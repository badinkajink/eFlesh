# eFlesh Visualizer

```pip install reskin-sensor anyskin pygame matplotlib numpy==1.26.4```

1. Connect the magnetometer circuit board to the microcontroller using the provided QWIIC cable.

2. Connect the microcontroller (Adafruit QT Py) to your computer using a USB-C cable.

3. Find the dev path/COM port your microcontroller is connected to:
<br><br> <b>Linux:</b> `ls /dev/ | grep -e ACM -e USB`. This is generally `/dev/ttyACM0` or `/dev/ttyUSB0`. <br>If you get a `can't open device "<port>": Permission denied` error, modify permissions to allow read and write on that port: `sudo chmod a+rw <port>`
<br> <b>MacOS:</b> `ls /dev/ | grep cu.usb`. This is generally `cu.usbmodem*`.
<br> <b>Windows:</b> Open Device Manager. Click `View` and select `Show Hidden Devices`. Locate `Ports (COM & LPT)`. Note the `COM` port corresponding to the QT Py.
<br><br>If you have no other devices connected, this should give you a single path. If you see multiple, disconnect the microcontroller and run the command again. Reconnect the microcontroller and re-run the command. The additional path is your `<port>`.