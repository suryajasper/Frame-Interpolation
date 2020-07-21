import {bilateralFilter} from "./bilateralFilter.js"
var arr = [];
var parent = document.getElementById('calcMeanSquaredErrorDiv');
document.getElementById('imageIn').onchange = function() {
  if (this.files && this.files[0]) {
    for (var file of this.files) {
      var reader = new FileReader();
      reader.onload = function(e) {
        var img = new Image();
        var bf = new bilateralFilter();
        bf.sigma = 4;
        bf.kernelsize = 4*bf.sigma; //95% of the data should fit into 4 sigma

        img.src = e.target.result;

        img.onload = function(){
          var cda = bf.run(img);
          var can = document.createElement('canvas');
          can.width = img.width;
          can.height = img.height;
          var ctx = can.getContext('2d');
          ctx.putImageData(cda,0,0);
          can.style.display = 'block';
          parent.appendChild(can);
          console.log(can);
          arr.push({ctx: can.getContext('2d'), width: can.width, height: can.height, base64: this.src});
        }
      }
      reader.readAsDataURL(file); // convert to base64 string
    }
  }
}
function base64torgb(base64, width, height) {
  var newCanvas = document.createElement('canvas');
  newCanvas.src = base64;
  return newCanvas.getContext('2d').getImageData(0, 0, width, height).data;
}
document.getElementById('calcMeanSquaredError').onclick = function(e) {
  e.preventDefault();
  console.log(arr);
  for (var obj of arr) {
    var sum = 0;
    var cartoon = obj.ctx.getImageData(0, 0, obj.width, obj.height).data;
    var orig = base64torgb(obj.base64, obj.width, obj.height);
    for (var i = 0; i < orig.length; i++) {
      sum += Math.pow(cartoon[i]-orig[i], 2);
    }
    sum /= orig.length;
    console.log(sum);
  };
}
