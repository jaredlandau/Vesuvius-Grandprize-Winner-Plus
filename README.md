# Vesuvius GP+

This repository is an extended implementation of the [Vesuvius Challenge 2023 Grand Prize winner](https://github.com/younader/Vesuvius-Grandprize-Winner). The primary goal of this repository is to greatly simplify the new user experience for those looking to get started with ink detection. This project also features some extra functionality compared to the original ink detection script, giving users fine control over exactly which layers they perform inference on. Read [this post](https://medium.com/@jaredlandau/vesuvius-challenge-ink-detection-part-1-introduction-1cb125a56b21) for a brief introduction to some of the experimentation I've done using this script.

<br/>

<div align="center">
  <img src="pictures/inference-example-n1n2n3n5n10.gif" alt="ink detection demo animation" width="350" />
  <p><i>Comparison of inferences performed on 1, 2, 3, 5, and 10 layers.</i></p>
</div>


## Installation

### WSL

If you're on Windows, I'd highly recommend using WSL (Windows Subsystem for Linux) to run the inference script -- it's by far the easiest way to get started.

Open PowerShell or Command Prompt in administrator mode, and then type the following command:

```
wsl --install
```

Once that's finished, you can choose a Linux distribution to use. Ubuntu is the default, so we'll stick with that.

```
wsl --install ubuntu
```

After this, you'll be prompted to set up an account by entering a username and password. Once you've done so, you can log in with the command below.

```
wsl --distribution ubuntu --user <username>
```

You can check how much memory is allocated to your WSL instance by typing `free` or `top`. If you'd like to allocate more memory, you can do so by creating an empty file named `.wslconfig` under your `C:/Users/name` directory. Ensure it doesn't have any file extension (such as .txt). Open the file, and paste in the text below:

```
[wsl2]
memory=8GB
```

You can change the amount of memory to whatever you'd like, depending on your hardware. Once you're finished, shut down and restart WSL for the changes to take effect.

```
wsl --shutdown
```
```
wsl --distribution ubuntu --user <username>
```

Running the `free` or `top` command again should now show your newly allocated memory.

### Downloading segment data

Before we run the script, we're going to need at least one segment downloaded to run inference on. You can find instructions on how to access the data [here](https://scrollprize.org/data).

You can download as many (or as few) layers as you'd like. Until you've had the chance to experiment a bit, I'd recommend starting with layers that are relatively small, between 50 MB and 250 MB each. Otherwise, each inference run could take a while to finish.

Your initial file structure should look something like this:

```
Vesuvius-Grandprize-Winner-Plus/
├─ scrolls/
│  ├─ 20230902141231/
│  │  ├─ 20230902141231_mask.png
│  │  ├─ layers/
│  │  │  ├─ 30.tif
│  │  │  ├─ 31.tif
│  │  │  ├─ 32.tif
├─ predictions/
│  ├─ 20230902141231/
├─ inference_timesformer.py
```

You should create a new folder in the 'scrolls' directory for each new segment you download. Each segment should have its own 'layers' folder and mask image. Inside the 'layers' folder, drop in as many layers as you need from any part of the segment.

You don't need to create any folders in the 'predictions' directory, the script will output inferences there automatically. Prediction filenames contain lots of useful information, see below:

```
20230902141231_prediction_n5_s29e33_20240601001122.png
```
Which means that this prediction is:
* on segment `20230902141231`
* `n5` on 5 layers
* `s29` starting at layer 29
* `e33` ending at layer 33
* and was generated with a timestamp of `20240601001122`

The timestamp is mostly there so that you can differentiate each inference you generate, but it also stops you from accidentally overwriting previous test runs.

### Running the script

Navigate to the mounted directory containing your installation of this repository.

```
cd /mnt/c/Users/name/folder/Vesuvius-Grandprize-Winner-Plus/
```

Run the below command to ensure you have all the required packages installed.

```
pip install -r requirements.txt
```

You can then run inference using the following command:

```
python3 inference_timesformer.py --segment_id <id> --start <s> --num_layers <n>
```

Where `--start` determines the starting layer for an inference run, and `--num_layers` determines how many layers above the starting layer are included. There are many other flags that can be set within the script to control settings like the batch size, stride, number of workers, etc.

### WandB

If this is your first time running the script, you may be prompted to set up a [WandB](https://wandb.ai/site) account. Setting up an account is quick and easy, and allows you to view your results and statistics on their online dashboard, so I'd highly recommend doing so.


## Changelog
* Added the option to select the number of layers to perform inference on
* Added the option to select the starting layer for an inference run
* Added detailed filenames to generated predictions, including layer and timestamp data
* Added detailed console outputs, so that users can easily track the progress of inference runs
* Fixed a persistent divide-by-zero error in inference_timesformer.py
* Refactored inference_timesformer.py
* Merged eval_scrolls and train_scrolls directories, to prevent data duplication
* Cleaned up home directory to simplify user experience


## Contact
Feel free to contact me on Discord if you have any questions at all (@tetradrachm).


## License

This repository is open-source code, licensed under the MIT License. See [LICENSE.txt](https://github.com/jaredlandau/Vesuvius-Grandprize-Winner-Plus/blob/main/LICENSE.txt) for details.


## Acknowledgments

Huge thanks to Youssef Nader, Luke Farritor, and Julian Schillinger for their fantastic Grand Prize submission, which this repository is directly built on.

And of course, a massive thanks to the organisers of the Vesuvius Challenge for their fantastic competition and invaluable dataset. I'd highly encourage people to join the community over at the [Discord](https://discord.gg/V4fJhvtaQn), it's a great place to start learning and contributing!
