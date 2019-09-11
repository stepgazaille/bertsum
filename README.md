## Implementation of 'Pretraining-Based Natural Language Generation for Text Summarization'

Paper: https://arxiv.org/pdf/1902.09243.pdf 

### Versions
* python 2.7
* PyTorch: 1.0.1.post2


### Setup
The quickest way to get up-and-running is to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and use the provided docker image:
```
# Pull docker image:
docker pull stepgazaille/bertsum

# Or, build docker image:
docker build -t stepgazaille/bertsum .
```
Given that this repository was cloned to ```~/bertsum```, the following command will provide access to a uSum docker container terminal:
```
docker run --runtime=nvidia -it --rm \
    -v $(realpath ~/bertsum):/home/bertsum \
    stepgazaille/bertsum \
    /bin/bash
```
Use the following commands if you must perform a "bare-metal" installation:
```
pip install -r requirements.txt
```

### Preparing package/dataset
1. Download chunk CNN/DailyMail data from: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail
2. Run: `python -m utils.data` to create pickle file that will be used in my data-loader

### Running the model
For me, the model was too big for my GPU, so I used smaller parameters as following for debugging purpose. 
`CUDA_VISIBLE_DEVICES=3 python main.py --cuda --batch_size=2 --hop 4 --hidden_dim 100`

### Note to reviewer:
* Although I implemented the core-part (2-step summary generation using BERT), I didn't have enough time to implement RL section. 
* The 2nd decoder process is very time-consuming (since it needs to create BERT context vector for each timestamp).