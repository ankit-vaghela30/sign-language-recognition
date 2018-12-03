/* 
@author: Gaurav Agarwal
@description: The main JS file.
@Date: 2nd Dec 2018. 
*/

var _streamCopy = null;     // used to stop the webcam video transmission
var screenshotInterval = null;

function hasGetUserMedia() {
    return !!(navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia);
}

/**
 * Checks whether the browser supports the getUserMedia() function for WebCam access.
 */
function checkBrowserSupport() {
    if (hasGetUserMedia()) {
        // Good to go!
        getWebcamFeed();
    } else {
        alert('-----Please update your browser-----\nCertain features are not supported by your browser!');
    }
}

/**
 * Displays the web cam feed on the webpage.
 */
function getWebcamFeed() {
    const constraints = {
        video: true
    };

    const video = document.querySelector('video');

    navigator.mediaDevices.getUserMedia(constraints).
        then((stream) => {
            _streamCopy = stream;
            video.srcObject = stream
        });

    const img = document.querySelector('#screenshot');
    const canvas = document.createElement('canvas');
    screenshotInterval = setInterval(function () {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        // Other browsers will fall back to image/png
        img.src = canvas.toDataURL('image/webp');
    }, 2000);
}

/**
 * Disables the web cam feed displayed on the webpage
 */
function disableWebcamFeed() {
    try {
        clearInterval(screenshotInterval);
        const img = document.querySelector('#screenshot');
        img.src = "";
        _streamCopy.stop(); // if this method doesn't exist, the catch will be executed.
    } catch (e) {
        _streamCopy.getVideoTracks()[0].stop(); // then stop the first video track of the stream
    }
}