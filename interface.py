from smt_trainer import SMT_Trainer
import torch

from math import sqrt
import gradio as gr
import numpy as np
import pandas as pd
import cv2
from numpy.typing import ArrayLike
from cv2.typing import MatLike

CA_layers = list()

colors = [  (128, 0, 0),
			(128, 64, 0),
			(128, 128, 0),
			(0, 128, 0),
			(0, 128, 128),
			(0, 64, 128),
			(0, 0, 128),
			(128, 0, 128),
			(128, 0, 0)
			]

def contrast(elem):
	return elem!=0

def overlay(background:np.ndarray, overlay:np.ndarray, alpha=1):
	"""
	:param background: BGR image (np.uint8)
	:param overlay: BGRA image (np.uint8)
	:param alpha: Transparency of overlay over background
	
	returns BGR image of combined images (np.float32)
	"""

	# add alpha channel to background
	background = np.concatenate([background, np.full([*background.shape[:2], 1], 1.0)], axis=-1 )

	# normalize overlay alpha channel from 0-255 to 0.-1.
	alpha_background = 1.0
	alpha_overlay = overlay[:,:,3] / 255.0 * alpha

	for channel in range(3):

		background[:,:,channel] = 	alpha_overlay * overlay[:,:,channel] + \
									alpha_background * background[:,:,channel] * ( 1 - alpha_overlay )

	background[:,:,3] = ( 1 - ( 1 - alpha_overlay ) * ( 1 - alpha_background ) ) * 255

	# ignore alpha channel because gradio doesnt care
	# also divide by 255 because somehow it needs a float image even though it gives int images
	return (background[:,:,:3]/255.0).astype(np.float32)

def generate_CA_images(token_idx, image, multiplier=1):

	global CA_layers

	CA_final_images = []

	# resize to fit input image (value in 0-1)
	masks = [ cv2.resize(CA_layers[layer_idx][token_idx],
							interpolation=cv2.INTER_NEAREST,
							dsize=(image.shape[1], image.shape[0])) for layer_idx in range(0, len(CA_layers)) ]

	# apply multiplier
	masks = [ mask*multiplier for mask in masks ]
	
	# normalize values above 1
	masks = [ mask/np.max(mask) if np.max(mask)>1 else mask for mask in masks ] 

	# (convert to values in 0-255)
	masks = [ np.round(mask*255.0).astype(np.uint8) for mask in masks ]


	# add singleton dimension as channel
	masks = [ np.expand_dims(mask, axis=-1) for mask in masks ]

	# base color + transparency mask = BGRA
	CA_token = [ np.concatenate((	np.full(shape=image.shape, fill_value=colors[i]), 
							  		masks[i]), 
									axis=-1) for i in range(9) ]                  

	# attention layers individually overlaid with input image
	CA_final_images = [ overlay(image, ca) for ca in CA_token ]

	return CA_final_images

def make_predictions(checkpoint, input_image):

	global CA_layers

	model_wrapper = SMT_Trainer.load_from_checkpoint(checkpoint)

	input_image = np.mean(input_image, axis=2, keepdims=True) # 3 channels to one
	input_image = np.transpose(input_image, (2,0,1))[None, :] # add batch size as well, [B, C, H, W]
	input_image = torch.from_numpy(input_image).to(device=model_wrapper.model.positional_2D.pe.device)

	input_image = input_image.to(torch.float32)

	# width / height
	aspect_ratio = input_image.shape[3]/input_image.shape[2]

	# 8 attention layers * [channels | seq_len | extracted_features]
	#   extracted features is FLAT input_image shape divided by 16
	predicted_seq, predictions = model_wrapper.model.predict(input_image)

	# seq_len | reduced_h * reduced_w
	CA_layers = [ ca_layer.squeeze() for ca_layer in predictions.cross_attentions ]
	
	seq_len = CA_layers[0].shape[0]
	att_w = round(sqrt(CA_layers[0].shape[1] * aspect_ratio))
	att_h = round(sqrt(CA_layers[0].shape[1] / aspect_ratio))

	# make the attention 2-D
	CA_layers = [ att.reshape( seq_len, att_h, att_w ) for att in CA_layers ]

	# convert to numpy
	CA_layers = [ att.cpu().detach().numpy() for att in CA_layers ]
	# ^^^ we store this, then generate the actual images to display ONLY whenever the token slider is moved

	## overlay all of them as overall attention
	overall = np.empty(CA_layers[0].shape)
	for ca in CA_layers:
		overall += ca

	## normalize
	overall /= np.max(overall)

	CA_layers.append(overall)

	#return "\t".join(predicted_seq)
	return pd.DataFrame(predicted_seq)

def test():
	print("called test")

def save_selected_tab(index):

	print(f"index: {index}")
	return index

def define_interface():

	# main components
	predicted_seq_output = gr.Textbox(label="Predicted Sequence", interactive=False)
	image_input = gr.Image(label="Input Image")
	tabs = gr.Tabs()
	
	# knob components
	token_slider = gr.Slider(minimum=0, maximum=0, step=1, 
							label="Pick a token", 
							info="Select a predicted token to visualize the attention it pays in the input sample",
							visible=False)

	intensifier_slider = gr.Slider(minimum=1, maximum=10, step=1,
									label="Intensify attention",
									info="Use this slider to intensify the attention values to better see differences",
									visible=False)


	# table
	token_table = gr.DataFrame(interactive=False)

	with gr.Blocks() as page:

		gr.Markdown("# SMT Demonstrator")

		with gr.Row():

			with gr.Column():

				#predicted_seq_output.render() #= gr.Textbox(label="Predicted Sequence", interactive=False)
				#predicted_seq_output = gr.Numpy(label="Predicted sequence", headers=["a", "b"], value=["c", "d"], column_widths="80px" )
				#predicted_seq_output.style
				image_input.render()
				#token_table.render()

				gr.Interface(make_predictions,
							inputs=[gr.File(label="Model Checkpoint File"),
									image_input],
							#outputs=[predicted_seq_output],
							outputs=[token_table],
							flagging_mode='never')
	

			with gr.Column(scale=2):
				

				token_table.change( ( lambda tokens : (gr.Slider(maximum=tokens.shape[0], visible=True), gr.Slider(visible=True)) ),
					   				inputs=[token_table],
									outputs=[token_slider, intensifier_slider] )

				token_slider.render()

				tab_selected = gr.Number(value="8", visible=False) # on Overall Attention tab by default
				
				# genera las imagenes cada vez que se mueve el slider
				@gr.render( inputs	=[token_table, token_slider, image_input, intensifier_slider, tab_selected],
							triggers=[token_slider.release, intensifier_slider.release] )
				def render_images_display(prediction, slider, image, intensifier, tab_no):

					if prediction.shape[0] > 0:

						images = generate_CA_images(slider, image, intensifier)

						gr.Markdown(value="## Contents of the Cross-Attention layers")

						print(f"tab_no: {tab_no}")

						with gr.Tabs(selected=f"{tab_no}") as tabs:

							with gr.Tab(f"Overall", id="8") as tab_overall:
								tab_overall.select( (lambda : gr.Number(8)), outputs=[tab_selected] )
								gr.Image(value=images[8])

							with gr.Tab(f"Layer 1", id=f"0") as tab_1:
									tab_1.select( (lambda : gr.Number(0)), outputs=[tab_selected] )
									gr.Image(value=images[0])

							with gr.Tab(f"Layer 2", id=f"1") as tab_2:
									tab_2.select( (lambda : gr.Number(1)), outputs=[tab_selected] )
									gr.Image(value=images[1])

							with gr.Tab(f"Layer 3", id=f"2") as tab_3:
									tab_3.select( (lambda : gr.Number(2)), outputs=[tab_selected] )
									gr.Image(value=images[2])

							with gr.Tab(f"Layer 4", id=f"3") as tab_4:
									tab_4.select( (lambda : gr.Number(3)), outputs=[tab_selected] )
									gr.Image(value=images[3])

							with gr.Tab(f"Layer 5", id=f"4") as tab_5:
									tab_5.select( (lambda : gr.Number(4)), outputs=[tab_selected] )
									gr.Image(value=images[4])

							with gr.Tab(f"Layer 6", id=f"5") as tab_6:
									tab_6.select( (lambda : gr.Number(5)), outputs=[tab_selected] )
									gr.Image(value=images[5])

							with gr.Tab(f"Layer 7", id=f"6") as tab_7:
									tab_7.select( (lambda : gr.Number(6)), outputs=[tab_selected] )
									gr.Image(value=images[6])

							with gr.Tab(f"Layer 8", id=f"7") as tab_8:
									tab_8.select( (lambda : gr.Number(7)), outputs=[tab_selected] )
									gr.Image(value=images[7])

							

						#tabs.select(  )
					return
				
				intensifier_slider.render()

	return page

if __name__=="__main__":
	page = define_interface()
	page.launch(share=False)

'''
with gr.Blocks() as demo: 
	radio = gr.Radio([1, 2, 4], label="Set the value of the number") 
	number = gr.Number(value=2, interactive=True) 
	radio.change(fn=lambda value: gr.update(value=value), inputs=radio, outputs=number) 
demo.launch()
'''