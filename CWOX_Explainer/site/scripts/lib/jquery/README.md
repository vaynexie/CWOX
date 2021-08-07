# d3-rect

draw rectangles with path elements

## Overview

[d3-shape](https://github.com/d3/d3-shape/) provides handy tools which help with drawing many common shapes, but curiously enough, rectangles are not among them. In most situations it is sufficient to use a  [rect element](https://developer.mozilla.org/en-US/docs/Web/SVG/Element/rect), but rect elements do not have path strings, and thus cannot be interpolated into a different shape. This plugin is a tiny helper which draws your rectangles using [path elements](https://www.w3.org/TR/SVG/paths.html) so they can be [animated](https://github.com/d3/d3-transition).

## Installing

If you use NPM, `npm install d3-rect`. Otherwise, download the [latest release](https://github.com/vijithassar/d3-rect/releases/latest).

## API

**d3.rect** is a function which returns a path string when passed an object containing x, y, height, and width properties.

```js
// append a path element
d3.select('div').append('path')
    // set the path string
    .attr('d', function() {
        var dimensions,
            path_string;
        // specify the dimensions
        dimensions = {
            x: 10,
            y: 20,
            height: 100,
            width: 200
        };
        // calculate the path string from the dimensions
        path_string = d3.rect(dimensions);
        return path_string;
    });
```