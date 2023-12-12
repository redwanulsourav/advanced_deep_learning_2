# Simple Text Generator

I have made a character level text generator where the model is encoding each character. Taking a sequence of encoded characters, the model is generating the next character

## Testing the Model
To test the model, I have prepared a _test.py_. I have also generated a _best3.pt_ file after my own training. This is to ensure that the model can be tested quickly without training.

To train the model, run the following command

    python test.py -w [weights_path] -c [number of characters to generate]

For example, the following command will use the _best3.pt_ file in this repository and generate 1000 characters.

    python test.py -w best.pt -c 1000


## Training the Model

To train the model, download the text dataset file from the following URL

Dataset URL: [https://drive.google.com/file/d/1PJy1VmuPvEEre3VfoiCSatDW6VrTs1Ac/view?usp=sharing](https://drive.google.com/file/d/1PJy1VmuPvEEre3VfoiCSatDW6VrTs1Ac/view?usp=sharing)

- The dataset file, _internet_archive_scifi_v3.txt_ should be in the same directory as train.py
- To train the python library dependencies, run the following commands

        pip install -r requirements.txt
    
- To train the model, run the following command

        python train.py


- After the training a process, a sample generated text is printed
- After training process, the following files are produced

    - best.pt - it stores the latest trained model.
    - encode_dict.json
    - decode_dict.json
- _encode_dict.json_ and _decode_dict.json_ contains the character encodings for mapping.

## Testing the Model

