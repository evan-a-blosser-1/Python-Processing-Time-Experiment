# Processing Time Experiment

This is an experiment to measure the processing speed, and calcuation erors, of Python with various configurations. These configurations are:

1) The basic processing with a single Central Processing Unit (CPU) core
2)  Multiple CPU cores using multiprocessing
3)  Graphics Processing Unit (GPU) using tensorflow 

## How-To Run

To run this program simply download both the `Processing_Time_Exp.py` along with the provided Asteroid_Database folder, and run in any Interactive Development Environment

*Note: The Meshgrid is initially set to the maximum of 10,000 & all the CPU cores will be used!! This can be modified below the '__main__' definition*

### Don't have tensorflow-gpu?
regular TensorFlow leverages the CPU for calculations, you MUST use tensrflow-gpu, and select a supported Graphics Processing Unit (GPU)
1) pip install both tensorflow and tensorflow-gpu
2) Download supporting CUDA & cuDNN software releases.
3) Run and collect data 
