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

### Reading data

For easy access to the data stored in the dragon .data format
we provide the `dragonboard.File` interface.

To loop over all events in a file, use:
```{python}
import dragonboard

with dragonboard.File('Ped444706_1.dat') as ped_file:
    for event in ped_file:
        # do something awesome
```

If you want to read everything into memory, you can use the `.read`
method:

```{python}
with dragonboard.File('Ped444706_1.dat') as ped_file:
    events = ped_file.read()

print(len(events))
```

If you just want a few events to test yout code, provide 
the `max_events` keyword:

```{python}
with dragonboard.File('Ped444706_1.dat', max_events=10) as ped_file:
    events = ped_file.read()

print(len(events))
```

An event is currently just a `collections.namedtuple` containing
a header, the roi and the data.

As default, data is a `numpy` structured array with two columns for
the different gains (`high` and `low`) and 8 rows for every pixel.
In each cell is a `roi`-length array with the raw data.

To plot the `high`-gain channel of pixel 3 of the first event you could do:


```{python}
import dragonboard
import matplotlib.pyplot as plt


with dragonboard.File('Ped444706_1.dat') as ped_file:
    event = ped_file[0]

plt.plot(event.data['high'][3])
plt.show()

```





### DragonViewer

Use the `dragonviewer` executable to view some data, it is in your `$PATH` after
you installed this module.

`dragoviewer [<inputfile>]`

If you do not provide `<inputfile>`, a file dialog will open

![Alt text](/dragonviewer.png?raw=true "Optional Title")
