# Solving Taxi Environment

This project focuses on solving the Taxi-v3 environment from OpenAI Gym using reinforcement learning techniques.


## Installation

### Prerequisites:
- Python 3

### Install Dependencies:
1. Clone this repository.
2. Navigate to the project directory.
3. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

There are two main ways to run this project:

### 1. Locally:
1. Ensure the dependencies are installed (see above).

### 2. Google Colab:
1. Create a new Colab notebook.
2. Upload the project files to your notebook.
3. Install the additional packages required for visualization:

    ```python
    !pip install IPython
    ```

4. Run the desired scripts within the notebook environment.

## Visualization (Google Colab Only)

The provided code offers functionalities for visualizing the agent's performance as a video. These functions are particularly useful in Colab:

### Visualization Functions:

```python
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from IPython import display as ipythondisplay
from IPython.display import HTML
import base64
import io

def display_video(file):
    mp4 = file
    video = io.open(mp4, 'r+b').read()
    encode = base64.b64encode(video)
    ipythondisplay.display(ipythondisplay.HTML(data='''<video alt="test" autoplay
                                                   loop controls style="height: 400px;">
                                                   <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                                                   </video>'''.format(encode.decode('ascii'))))   

def frames_to_video(frames, output_path, fps=5):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264")
    display_video(output_path)
```

### These functions allow you to

Convert a sequence of frames to a video file.
Display the video directly in the Colab notebook.

### How to Use:

Save the frames of your Taxi environment during training or evaluation.
```python
Call frames_to_video(frames, 'output.mp4')
```

## Output Videos

You can find example output videos demonstrating the performance of our Q-learning and SARSA algorithms in the `assets` folder:

These videos showcase the agent's behavior in the Taxi-v3 environment after training with each algorithm. You can observe how the agent navigates the environment, picks up passengers, and drops them off at their destinations.

To view these videos:
1. Clone the repository
2. Navigate to the `assets` folder
3. Open the video files with a media player that supports MP4 format

Note: If you're viewing this README on GitHub, the videos may not play directly. In that case, you'll need to download the repository and view the videos locally.



## Features
*  Implementation of Q-learning and SARSA algorithms for Taxi-v3 environment.
* Interactive visualizations (for Google Colab) to track learning progress and performance metrics.
* Customizable hyperparameters (e.g., learning rate, discount factor) to experiment with different RL settings.



## Contributing
Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License
This repository is licensed under the MIT License. See the LICENSE file for more details   [MIT License](LICENSE).
