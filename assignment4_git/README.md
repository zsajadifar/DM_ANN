# ###########################################################################
# ###########################################################################
# #### DATA MINING AND NEURAL NETWORKS                                   ####
# #### Prof. Dr. ir. Johan A. K. Suykens                                 ####
# #### Assignment 4 - Instructions                                       ####
# ###########################################################################
# ###########################################################################


# Instructions for running python demos on colab

## Steps for the demo of BatchNormaliztion exercise:

1. Go to `https://colab.research.google.com/`, login with your google account and start a `New Notebook`
2. Upload BatchNormalization.zip file on colab and start a hosted runtime by clicking `connect` on the top-right corner.
3. Use the following command to unzip the files: ` !unzip BatchNormalization.zip `. Basically copy-paste the command in the cell and click the `play` button on the left-side of cell.
4. Use the following command to run the main file: ` !python BatchNormalization.py `

## Similarly for the Demo of Variational AutoEncoder:
1. Repeat steps 1 and 2 from above.
2. Use the following command to unzip the files: ` !unzip VariationalAutoEncoder.zip `.
3. Run the following commands:

```
%tensorflow_version 1.x
!python vae_pytorch.py

``` 

The figures will be saved on the colab's directory. You can later download them for presenting in the report. 

