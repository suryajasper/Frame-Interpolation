document.getElementById('imageIn').onchange = function() {
  if (this.files && this.files[0]) {
    for (var file of this.files) {
      var reader = new FileReader();
      reader.onload = function(e) {
        if (window.localStorage.getItem('images') !== null) {
          var images = JSON.parse(window.localStorage.getItem('images'));
          images.push(e.target.result);
          window.localStorage.setItem('images', JSON.stringify(images));
        } else {
          window.localStorage.setItem('images', JSON.stringify([e.target.result]));
        }
      }
      reader.readAsDataURL(file); // convert to base64 string
    }
  }
}

document.getElementById('calcMeanSquaredError').onclick = function() {
  var images = JSON.parse(window.localStorage.getItem('images'));
  for (var imageSRC of images) {
    var image = new MarvinImage();
    image.load(imageSRC, function() {
      var canvas = document.createElement('canvas');
      canvas.width = image.getWidth();
      canvas.height = image.getHeight();
    })
  }
}
