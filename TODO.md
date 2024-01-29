
 - Allow configuring whether to use True/False token or the following period as the sentence representation.
 - Save hidden states of all layers by default, and select layer to probe separately.


 - Create step that combines the training data from the different data configurations in order to hopefully debias the directions learnt by the various methods.
   - call it 'MixData' and have arguments:
     - list of data sources
     - list of weights of that data source in the final mixed data


 - Create/Modify a generation step that moves a token along a specified direction in a specified layer.
