# Templates

## How to implement your algorithms
To adapt your methods to this template you can adapt the _inference.py_ script. The script contains two methods _infer\_ct_ and _infer\_mr_, here you can place your inference code. If your method is modality agnostic you can simply copy the inference code or make the second method call the first one. If you are more comfortable with coding you can also change the mechanism to only require one inference method.
See the comments in the methods in _inference.py_ for further instructions in place.

## Handling Dependencies

Place the requirements of your algorithms in the respective _requirements.txt_ files.
To integrate your models, load weights or decision tree configurations from _./models/_.
Or you can put your code directly into the image, in this case remember to include your custom files in the _Dockerfile_.

## Testing Locally

The templates also come with some bash scripts to check if your container works as GrandChallenge would run them.
  - *do_build.sh*: Builds the container under the name:tag _topaneu-26-task[1/2]:latest_.
  - *do_save.sh*: Exports the container image under the name *topaneu-26-task[1/2]_YYYY-MM-DD_hh-mm-ss.tar.gz* such that you can upload it to GC.
  - *do_test_run.sh*: Builds and runs the container as GC would. **NOTE** This script is aimed at running in an environment where GPU is available. If you're in a ressource constrained environment remove ```--gpus all``` from the run command in line 90.

To use the scripts navigate to the respective template directory:
```bash
cd templates/task[1/2]
```
Then to run the scripts do:
```bash
bash [do_build/do_save/do_test_run].sh
```