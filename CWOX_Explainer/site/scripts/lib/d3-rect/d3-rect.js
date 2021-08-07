(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (factory((global.d3 = global.d3 || {})));
}(this, function (exports) { 'use strict';

    // generate path string based on input parameters
    var rect;

    rect = function(dimensions) {
        var required,
            x_offset,
            y_offset,
            upper_left,
            upper_right,
            lower_right,
            lower_left,
            close,
            path_string;
        required = dimensions.height > 0 && dimensions.width > 0;
        if (! required) {
            console.error('rectangle path generator requires both height and width properties');
            return;
        }
        x_offset = dimensions.x || 0;
        y_offset = dimensions.y || 0;
        upper_left = 'M ' + x_offset + ',' + y_offset;
        upper_right = 'l ' + dimensions.width + ',0';
        lower_right = 'l 0,' + dimensions.height;
        lower_left = 'l ' + dimensions.width * -1 + ',0';
        close = 'z';
        path_string = [
            upper_left,
            upper_right,
            lower_right,
            lower_left,
            close
        ].join(' ');
        return path_string;
    };

    var rect$1 = rect;

    exports.rect = rect$1;

    Object.defineProperty(exports, '__esModule', { value: true });

}));