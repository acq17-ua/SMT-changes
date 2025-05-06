from smt_trainer import SMT_Trainer
import gradio as gr
import torch
import numpy as np
from math import sqrt
import cv2

CA_layers = []

colors = [  (0.0,   0.0,   0.0),
			(0.0,   0.0,   0.0),
			(0.0,   0.0,   0.0),
			(0.0,   0.0,   0.0),
			(0.0,   0.0,   0.0),
			(0.0,   0.0,   0.0),
			(0.0,   0.0,   0.0),
			(0.0,   0.0,   0.0),
			]

def contrast(elem):
	return elem!=0

def make_predictions(checkpoint, input_image):

	global CA_layers
	global CA_final_images
	global input_image_h
	global input_image_w

	#print(f"type of checkpoint: {type(checkpoint)} of image: {type(input_image)}")

	model_wrapper = SMT_Trainer.load_from_checkpoint(checkpoint)

	input_image = np.mean(input_image, axis=2, keepdims=True) # 3 channels to one
	input_image = np.transpose(input_image, (2,0,1))[None, :] # add batch size as well, [B, C, H, W]
	input_image = torch.from_numpy(input_image).to(device=model_wrapper.model.positional_2D.pe.device)

	input_image = input_image.to(torch.float32)

	#print(f"initial input shape: {input_image.shape}")

	# width / height
	aspect_ratio = input_image.shape[3]/input_image.shape[2]

	# 8 attention layers * [channels | seq_len | extracted_features]
	#   extracted features is FLAT input_image shape divided by 16
	predicted_seq, predictions = model_wrapper.model.predict(input_image)

	#print(f"initial CA shape: {predictions.cross_attentions[0].shape}")
	#print(f"seq len: {len(predicted_seq)}")

	# seq_len | reduced_h * reduced_w
	CA_layers = [ ca_layer.squeeze() for ca_layer in predictions.cross_attentions ]
	
	input_image_h, input_image_w = CA_layers[0].shape[0], CA_layers[0].shape[1]

	seq_len = CA_layers[0].shape[0]
	att_w = round(sqrt(CA_layers[0].shape[1] * aspect_ratio))
	att_h = round(sqrt(CA_layers[0].shape[1] / aspect_ratio))

	print(f"image dims: {input_image.shape}")

	# make the attention 2-D
	CA_layers = [ att.reshape( seq_len, att_h, att_w ) for att in CA_layers ]

	# convert to numpy
	CA_layers = [ att.cpu().detach().numpy() for att in CA_layers ]
	# ^^^ we store this, then generate the actual images to display ONLY whenever the token slider is moved

	# deberia generar las imagenes del primer token
	CA_final_images = [ cv2.resize(CA_layers[layer_idx][0],
								   interpolation=cv2.INTER_NEAREST,
								   dsize=(input_image.shape[3], input_image.shape[2])) for layer_idx in range(len(CA_layers)) ]

	CA_final_images.append(np.zeros((256,256)))

	## overlay all of them

	"""     
	overall_attention = np.zeros(input_image.shape[3], input_image.shape[2])
	for i in range(len(CA_final_images)-1):

		overall_attention += cv2.addWeighted(cv2.bitwise_not(CA_final_images[i]),0.1,
											cv2.bitwise_not(CA_final_images[i+1]), 0.1 )
	"""
	#print(f"len ca_layers: {len(CA_layers)}")
	#print(f"final images len: {len(CA_final_images)}")
	#print(f"shape after resize: {CA_final_images[0].shape}")

	return predicted_seq

def define_interface():

	CA_tabs = []
	predicted_seq_output=None
	checkpoint_input=None
	image_input=None
	predict_button = None

	with gr.Blocks() as page:

		gr.Markdown("# Cross Attention Viewer")

		with gr.Row():

			with gr.Column():

				predicted_seq_output = gr.Textbox(label="Predicted Sequence", interactive=False)
				image_input = gr.Image(label="Input Image")

				gr.Interface(make_predictions,
							inputs=[gr.File(label="Model Checkpoint File"),
									image_input],
							outputs=[predicted_seq_output],
							flagging_mode='never')
				

			with gr.Column(scale=2):

				def generate_CA_images(token_idx, image):

					if image_input is None:
						return

					CA_final_images = []

					# resize to fit input image
					masks = [ cv2.resize(CA_layers[layer_idx][token_idx],
											interpolation=cv2.INTER_NEAREST,
											dsize=(image.shape[1], image.shape[0])) for layer_idx in range(len(CA_layers)) ]

					# add channels dimension
					masks = [ np.expand_dims(mask, axis=-1) for mask in masks ]

					# non-zero = 1, strength of attention will be shown through transparency
					masks = [ np.apply_along_axis(contrast, 2, ca) for ca in masks ]

					# base color + transparency mask = RGBA
					CA_token = [ np.concatenate((np.full(shape=image.shape, fill_value=colors[i]), masks[i]), axis=-1) for i in range(8) ]                  

					# individual layers overlaid with input image
					CA_final_images = [ cv2.addWeighted(np.concatenate((image, 
														 				np.ones((image.shape[0], image.shape[1], 1))), 
																		axis=-1), 
														1, 
														ca, 
														0.4, 
														0) for ca in CA_token ] 

					# overall view
					overall = np.concatenate((image, 
											np.ones((image.shape[0], image.shape[1], 1))), 
											axis=-1)
					
					for ca in CA_final_images:
						overall = cv2.addWeighted(overall, 1, ca, 0.125, 0)

					CA_final_images.append(overall)

					cv2.imshow("image", CA_final_images[8])

					with gr.Tab(f"Overall"):
						gr.Image(value=CA_final_images[8])

					for i in range(8):  
						with gr.Tab(f"Layer {i+1}") as tab:
							gr.Image(value=CA_final_images[i])
							CA_tabs.append(tab)
					return

				@gr.render(predicted_seq_output)
				def render_knobs(predicted_seq):

					print(f"rendering textbox ca_layer")

					if( predicted_seq is not None and len(predicted_seq)==0 ):
						print(f"textbox is empty ca_layer")
						return

					gr.Markdown(value="## Contents of the Cross-Attention layers")

					with gr.Tab(f"Overall"):
						gr.Image(interactive=False)

					for i in range(8):  
						with gr.Tab(f"Layer {i+1}") as tab:
							gr.Image(value=CA_final_images[i])
							CA_tabs.append(tab)

					if( predicted_seq is not None and len(predicted_seq)==0 ):
						print(f"textbox is empty slider")
						return
					
					slider = gr.Slider(minimum=0,
										maximum=CA_layers[0].shape[0],
										label="Pick a token",
										info = "Pick a token to see the attention it paid to each pixel in the image",
										step=1)

					slider.release(generate_CA_images, inputs=[slider, image_input])

				#@gr.render(inputs=[token_slider, image_input], triggers=[token_slider.release])

	return page

if __name__=="__main__":
	page = define_interface()
	page.launch(share=False)

