var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => res.sendFile(path.join(__dirname + '/public/index.html')));
app.use('/users', require('./routes/users'));
app.use('/faces', require('./routes/faces'));

// catch 404 and forward to error handler
app.use(function (req, res, next) {
    next(createError(404));
});

// error handler
app.use(function (er, req, res, next) {
    res.locals.message = er.error; // set locals, only providing error in development
    res.locals.error = req.app.get('env') === 'development' ? er : {};
    console.error("ErrorHandler", er)
    const status = er.status || 520 // 520 Unknown Error
    res.status(status).send({ status, error: er.error })
});

module.exports = app;
