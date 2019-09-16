# Instructions 
Requires Mcrae and CSLB data norms. The format of the data must be a csv placed in the data file in the form similar to the feature_matrix.dat file in the CSLB dataset.
Any future norming datasets or augmented datasets in this form should work given this format. 

## Mcrae Propery Norms
The norms can be found at https://sites.google.com/site/kenmcraelab/norms-data under "the he concepts with the features and numerous measures on both the concepts themselves, the features themselves, and concept-feature relations".
Download this file and use the utils.py build_norms function to format data. 

## CSLB Property Norms 
The norms can be found at http://www.csl.psychol.cam.ac.uk/propertynorms/ which will require further information to download.
These properties simply need converting to a csv file (although a .dat file may also work).

## Property Norm Info 
Each dataset should consist of rows of concepts with columns representing properties. The values are the frequency productions for which only properties 5 or greater counts are selected.
