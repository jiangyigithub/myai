
1. Unzip MobaXterm
2. Click "Session" and choose SSH 
Remote host: e.g. abts55134.de.bosch.com
3. mkdir xxx to create your own folder
4. cd xxx to go into your folder (using <Tab> to quickly complete your folder name)
cd .. to go out
5. Drag or click upload on left document view to add related files
6. module purge to clear loaded module
7. module load conda cudnn (use module list to check your current modules)
8. conda create -n XXX python=3.8 pandas
9. conda activate myEnv （use conda env list to check your current envs)
10. conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch (pip list to check the modules you installed)
11. python (to get into python env)
12. import torch
13. torch.cuda.is_available() (if return true, then cuda could be used for training)
14. exit() to quit python env
15. open mine.bsub from left side document view to change related config
16. bsub < mine.bsub to submit your job
17. bjobs to check it's status (using bjobs -l -u all -m abts55XXX | grep "gpus=" to check your assigned GPU)
18. bkill XXX (JOBID) to kill your job
19. How to use traffic_det_vgg16_demo.py ? (it's already working) 
	a. Set basic config e.g. classes, pretrained or not, paths 
	b. Data loader and preprocessing
	c. Using or Creating your own network
	d. If skip c step then choose an opensource model from torchvision.models (torchvision.models.<tab> to choose one)
	e. using pretrained weight, you could directly choose pretrained or not in step d. if you are using your own model you need to load a weight of yourself (vgg16.pkl in attachment is a weight file)
	f. (optional) set locked layer if needed in training
	g. Loss, Optimizer, Scheduler choose if needed
	h. Train model (not need if you only to use pretrained weight)
	i. Test model (predict and output result)
	j. Save to file 
	(comment all the .cuda code if you are using CPU to do inference)
20. Submit your result in AI platform