A file named README.txt that contains basic instructions on running your code, including what libraries, languages, etc., (including versions) to use, as well as what commands to use to run your code. This doesn’t need to be extremely detailed as long as we can successfully run your code based on your instructions.

Code Language:  Python

List of code files:
  stochastic_motion_textures.py (main function)
  layer_inpainting.py
  layer_motion.py

Requirements:
  cv2
  numpy
  scipy
  skimage
  numba

To run enter the following command, order of input matters:
python stochastic_motion_textures.py image.jpg layer_1_mask.jpg layer_1_type ... layer_n_mask.jpg layer_n_type number_of_frames

  where:
    image.jpg is the original image to be animated
    layer_1_mask.jpg is the foreground mask for the first foreground layer
    layer_1_type is the type of motion for the first foreground layer
    ...
    layer_n_mask.jpg is the foreground mask for the top most foreground layer
    layer_n_type is the type of motion for the top most foreground layer
    number_of_frames is the integer number of frames to render
        (code guarantees a fully looping set of images based on number of frames)

To generate an animated gif:
  ffmpeg -i image_directory/file_prefix_%04d.png out_video.gif


Special notes:
  Code is hard coded to use 3 letter extensions for images
  Order of layers matters, first layer is the bottom most foreground element,
    sequentially stacking to the last, top most, foreground element.
  Motion Types are as follows: boat, plant, water, cloud

Examples:
  python stochastic_motion_textures.py boatstudio.jpg boatstudio_water_mask.jpg water boatstudio_boat_mask.jpg boat 60
  python stochastic_motion_textures.py puerto_rico_boats.jpg puerto_rico_boats_left.jpg boat puerto_rico_boats_right.jpg boat 60
  python stochastic_motion_textures.py jet.jpg jet_mask.jpg cloud 60
  python stochastic_motion_textures.py bobble_head.jpg bobble_head_mask.jpg boat 30
