# InputImageSlicer
If you have labelled images that are too big to as an input for your CNN, this script will slice the image into multiple subimages.
The input parameters of the slice function are:
- File_name
- Input shape for the CNN (2 value tuple)
- Pandas dataframe of the labels with an x, y, w, h column

The outputs of the script are two pandas dataframes:
- A dataframe which contains the sliced imagename and where the image was sliced at
- A dataframe which contains the new label coordinates

For example, the following image:
![Labelled image](https://github.com/brianbarbieri/InputImageSlicer/blob/master/images/full.png)

Will be sliced into multiple images, for example of size (418, 418), as shown below:
![Sliced image](https://github.com/brianbarbieri/InputImageSlicer/blob/master/images/sliced.png)


The scipt is includes both a .py file and a notebook file.
