var canvas = document.getElementById('paint');
var ctx = canvas.getContext('2d');

let sketch = document.getElementById('sketch');
var sketch_style = getComputedStyle(sketch);
canvas.width = 28;
canvas.height = 28;

var mouse = {x: 0, y: 0};

/* Mouse Capturing Work */
canvas.addEventListener('mousemove', function(e) {
    mouse.x = e.pageX/10;
    mouse.y = e.pageY/10;
}, false);

/* Drawing on Paint App */
ctx.lineJoin = 'round';
ctx.lineCap = 'round';

ctx.strokeStyle = "white";
function getColor(colour){ctx.strokeStyle = colour;}
function getSize(size){ctx.lineWidth = size;}

canvas.addEventListener('mousedown', function(e) {
    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);

    canvas.addEventListener('mousemove', onPaint, false);
}, false);

canvas.addEventListener('mouseup', function() {
    canvas.removeEventListener('mousemove', onPaint, false);
    sendData(ctx.getImageData(0, 0, 28, 28));
}, false);

var onPaint = function() {
    ctx.lineTo(mouse.x, mouse.y);
    ctx.stroke();
};

function sendData(imageData){
    let data = [];
    
    for (let i=0; i<imageData.data.length; i++){
        if (i % 4 === 0){
            data.push(imageData.data[i]);
        }
    }

    setResult('');
    fetch(
    "https://localhost:5001/recognize", 
    {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(arrayToBase64String(data))
    }).then(res=>{
        return res.text().then(text=>setResult(text));
    });
}

function arrayToBase64String(a) {
    return btoa(String.fromCharCode(...a));
}

function setResult(result){
    document.getElementById('result').innerText = result;
}