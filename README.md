# Green Screen Background Replacement with Machine Learning and OpenCV

This project demonstrates a real-time green screen (chroma key) background replacement using OpenCV and machine learning techniques. The program dynamically detects and removes the green screen from a foreground video, replacing it with another video or image. The green screen detection is powered by HSV color space manipulation and K-Means clustering for optimal performance.

## Features

- **Real-Time Green Screen Detection**: Automatically detects and replaces the green screen background in a video using HSV color space.
- **Machine Learning (K-Means Clustering)**: Automatically finds the green color range using K-Means clustering for improved detection accuracy across different videos.
- **Manual HSV Range Adjustment**: Trackbars are provided for real-time fine-tuning of HSV values to handle different lighting conditions and shades of green.
- **Morphological Mask Refinement**: Uses morphological operations to clean up the mask and remove noise or inconsistencies.
- **Background Video Replacement**: Replaces the detected green screen with another video or static image for dynamic video compositing.

## Demo

[Green Screen Replacement Demo](https://youtu.be/bdedqKFqRp4?si=vbG8CKaf5pC2lq5q)
 <!-- Replace with an image or GIF showing the project in action -->

# How It Works
* HSV Conversion: The input video frames are converted from BGR to HSV color space, which simplifies color detection.
* K-Means Clustering: A machine learning approach is used to automatically determine the HSV range for the green screen by clustering similar colors.
* Manual Adjustment: You can manually adjust the HSV range using trackbars to accommodate different green screen shades or lighting conditions.
* Mask Creation: The green screen is isolated using a mask, and morphological operations like erosion and dilation are applied to refine the mask.
* Background Replacement: The green regions in the video are replaced by a background video or image, creating the final composited result.

##Contributing
Feel free to submit issues or pull requests. Contributions to improve the green screen detection or enhance the background replacement logic are welcome!

##License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Jatly/Green-Screen-Background-Replacement-with-Machine-Learning-and-OpenCV/blob/main/LICENSE) file for more details.
