ColdNAS: Search to Modulate for User Cold-Start Recommendation. WebConf 2023
## Usage:
## Data source:

    - For Last.FM, we use the data provided by TaNP (https://github.com/IIEdm/TaNP).(Already provided in .data.zip)

    - For MovienLens-1M, we use the data provided by MeLU (https://github.com/hoyeoplee/MeLU).

    - For BookCrossing, we use the data downloaded from (http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

## Requirements: 
     --python 3.7.0 --torch 1.7.1 --cuda 11.0 --numpy 1.19.3

## Run on Last.FM:
    - Unzip data 'data.zip'

    - Search by running 'search.py'

    - Select top-K by output alpha and change the model in ‘model_lfm4.py’

    - Train and evaluate the searched model by running 'evaluate.py'
    
