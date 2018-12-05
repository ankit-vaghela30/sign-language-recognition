/* 
@author: Gaurav Agarwal
@description: The main JS file.
@Date: 2nd Dec 2018. 
*/

var _streamCopy = null;     // used to stop the webcam video transmission
var screenshotInterval = null;
var frame_time = 1000;

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

    const canvas = document.createElement('canvas');
    screenshotInterval = setInterval(function () {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        // Other browsers will fall back to image/png
        // img.src = canvas.toDataURL('image/webp');
        // document.querySelector('#client_images').src = canvas.toDataURL('image/webp');
        sendImgToDetect(canvas.toDataURL('image/jpeg'));
    }, frame_time);
}

/**
 * Disables the web cam feed displayed on the webpage
 */
function disableWebcamFeed() {
    try {
        clearInterval(screenshotInterval);
        // document.querySelector('#screenshot').src = "";
        // $('#recognizedLetter').text("");
        _streamCopy.stop(); // if this method doesn't exist, the catch will be executed.
    } catch (e) {
        _streamCopy.getVideoTracks()[0].stop(); // then stop the first video track of the stream
    }
}

function resetFields() {
    disableWebcamFeed();
    // document.querySelector('#screenshot').src = "";
    $('#recognizedLetter').text("");
    $('#appendedText').text("");
    $('#accuracy').text("");
}

/**
 * Calls web service over ajax
 * @param {String} base64_img 
 */
function sendImgToDetect(base64_img) {
    base64_img = base64_img.replace('data:image/jpeg;base64,', '');
    var jsonQuery = {
        image: "" + base64_img
    }
    // console.log(base64_img);
    $.ajax({
        dataType: "text",
        url: "http://127.0.0.1:5000/upload",
        data: JSON.stringify(jsonQuery),
        method: 'POST',
        xhrFields: {
            withCredentials: true
        },
        crossDomain: true,
        contentType: 'application/json; charset=utf-8',
        // timeout: 50000,
        success: ajaxSuccess,
        error: ajaxFailure
    });
}

function ajaxSuccess(result) {
    result = JSON.parse(result);
    var acc = Math.round(result.accuracy);
    if(acc > 0){
        var letter = result.letter_detected.toUpperCase();
        $('#recognizedLetter').text(letter);
        $('#appendedText').append(letter);
    }
    $('#accuracy').text(acc + "%");
    // var imgDetect = result.hand_object;
    // imgDetect = imgDetect.substring(2, imgDetect.length - 1)
    // document.querySelector('#screenshot').src = imgDetect;
    // $('#screenshot').attr('src', imgDetect);
// "data:image/jpeg;base64," + 
    // console.log(imgDetect);
}

function ajaxFailure(jqXHR, textStatus, errorThrown) {
    disableWebcamFeed();
    console.log(jqXHR.status + " : " + textStatus + " : " + errorThrown);
}