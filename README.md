# Gesture Controlled Video Game Playing
We have aimed to build a game play system that allows users to interact with the game characters with just a regular camera and their hand.
We have built the system by first segmenting the hands using a Deep Residual UNet, and then by classifying it using a ResNet18. We have also tested the system on two games with slightly varying controls to check its versatility.
Apart from this, we are also presenting a segmented gesture dataset, with images of two kinds; multiple images of gestures for classification purposes, as well as segmented hand gestures for a dataset more suited to gesture recognition.

# Approach
The final system consists of 3 major modules: Hand Segmentation, Gesture Classification and Command Simulation. We trained a Deep Residual UNet for the hand segmentation and then separately trained a ResNet18 for gesture classification, and finally combined them with the noise suppression system and DirectX Scan codes in the command simulation module.

## Hand Segmentation
Based on the work by Zhang et al [1], a Deep Residual UNet was built to segment the hands in the incoming video feed. The model was trained for 50 epochs on EgoHands, HandOverFace and the Segmented Hands Dataset which we built using videos we shot of hands doing gestures. Each incoming frame from the video feed is passed through the UNet to get the segmented hand, which is then sent over for classification.

## Gesture Classification
Once each incoming frame is segmented, it is then passed through a ResNet18, so that it can be classified into one of 5 gestures. We have chosen 5 gestures, four represent the standard keys used for most games: up, down, left and right. A fifth ‘null’ gesture is used for when the player does not want to press any key. This is useful when the player must wait before moving or stop before carrying out a certain action. 
A pretrained ResNet18 is used with the last layer changed, to classify 5 gestures using a fastai framework. The model is trained for 3 epochs on the gesture dataset we created, along with further image augmentations. 

## Command Simulation
Once we have the classified gesture, we pass it through the noise suppression system and finally send in the corresponding command to the game. While the user switches between gestures, the classifier might classify the frames during the transition, which is noise. To overcome this, we have set up a buffer that stores the gestures classified for the last 10 frames. If a certain gesture occurs more than 5 times, it is selected as a command the user intentionally wants to pass and is sent over to the game. 
The command for each gesture is written in DirectX Scan codes. The codes are sent as integers to the game, which leads to the action corresponding to the command to be carried out in the game. 

# Results
The system was tested on two games: ‘Limbo’ and ‘Untitled Goose Game’. ‘Limbo’ is a famous platformer adventurer game created by Playdead, and we used the up, down, left, right and null commands to play this game.
‘Untitled Goose Game’ is a puzzle stealth game developed by House House. In this game, we switched up the commands a little, and instead switched the gesture for ‘down’ to ‘space-bar’, which when called causes the goose to honk. This was to test the versatility of the gestures and to see if they work well with other commands as well.
Demos of these game play sessions can be found in the 'Demos' folder.

# References
[1] Zhang, Zhengxin, Qingjie Liu, and Yunhong Wang. “Road Extraction by Deep Residual U-Net.” IEEE Geoscience and Remote Sensing Letters 15.5 (2018): 749–753. Crossref. Web.

[2] S. Bambach, S. Lee, D. J. Crandall and C. Yu, "Lending A Hand: Detecting Hands and Recognizing Activities in Complex Egocentric Interactions," 2015 IEEE International Conference on Computer Vision (ICCV), Santiago, 2015, pp. 1949-1957.

[3] Khan, Aisha Urooj and Ali Borji. “Analysis of Hand Segmentation in the Wild.” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018): 4710-4719

[4] Howard, Jeremy, and Sylvain Gugger. “Fastai: A Layered API for Deep Learning.” Information 11.2 (2020): 108. Crossref. Web.
