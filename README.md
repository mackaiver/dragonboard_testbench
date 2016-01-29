# dragonboard_testbench

A collection of programs to help with drs calibration of the LST test data of the dragon_cam.


## Installation

`pip install git+https://github.com/mackaiver/lst_calibration`

or, if you are developing

```
git clone http://github.com/mackaiver/lst_calibration
pip install -e lst_calibration
```

## Usage


### DragonViewer

Use the `dragonviewer` executable to view some data, it is in your `$PATH` after
you installed this module.

`dragoviewer [<inputfile>]`

If you do not provide `<inputfile>`, a file dialog will open

![Alt text](/dragonviewer.png?raw=true "Optional Title")
