var express = require('express');
var app = express();
var io = require('socket.io');
app.use(express.static(__dirname + '/client'));
app.use((req, res, next) => {
	res.setHeader('Access-Control-Allow-Origin', 'http://localhost:8000');
	res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
	res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type, Accept');
	next();
});

/*io.on('connection', function(socket) {
  socket.on('resemble', function(img1, img2) {
    var unirest = require('unirest');
  });
});*/

var http = require('http').createServer(app);
var port = process.env.PORT || 8000;
http.listen(port, function(){
  console.log('listening on 127.0.0.1/' + port.toString());
});
