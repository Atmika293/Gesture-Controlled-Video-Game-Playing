# Gesture Controlled Video Game Playing
We have aimed to build a game play system that allows users to interact with the game characters with just a regular camera and their hand.
We have built the system by first segmenting the hands using a Deep Residual UNet, and then by classifying it using a ResNet18. We have also tested the system on two games with slightly varying controls to check its versatility.
Apart from this, we are also presenting a segmented gesture dataset, with images of two kinds; multiple images of gestures for classification purposes, as well as segmented hand gestures for a dataset more suited to gesture recognition.

# Approach
The final system consists of 3 major modules: Hand Segmentation, Gesture Classification and Command Simulation. We trained a Deep Residual UNet for the hand segmentation and then separately trained a ResNet18 for gesture classification, and finally combined them with the noise suppression system and DirectX Scan codes in the command simulation module.

## Hand Segmentation
Based on the work by Zhang et al [1], a Deep Residual UNet was built to segment the hands in the incoming video feed. The model was trained for 50 epochs on EgoHands, HandOverFace and the Segmented Hands Dataset which we built. Each incoming frame from the video feed is passed through the UNet to get the segmented hand.

## Gesture Classification
Once each incoming frame is segmented, it is then passed through a ResNet18, so that it can be classified into one of 5 gestures. Each gesture corresponds to one key: up, down, left, right and null. The arrow commands are used for game play and the null command is a placeholder command. A pretrained ResNet18 is used with the last layer changed, to classify 5 gestures using a fastai framework. The model is trained for 3 epochs on the gesture dataset we created, along with further image augmentations. 

## Command Simulation
