const width = 1280;
const height = 960;
const margin = { top: 40, right: 110, bottom: 100, left: 40 };

const radius = 9;

var data_server = "http://0.0.0.0:5400"
var svg = d3.select("#canvas").attr("viewBox", [0, 0, width, height]);

var label2desc = {};
d3.json(data_server + "/json/label2desc.json").then(function (d) {
  label2desc = d;
  label2desc.others = "the rest classes";
});

var label2name = {};
d3.json(data_server + "/json/label2name.json").then(function (d) {
  label2name = d;
  label2name.others = "others";
});

var repr = {};
d3.json(data_server + "/json/repr.json").then(function (d) {
  repr = d;
  repr.others = "";
});

svg
  .append("text")
  .attr("x", width - margin.right + 10)
  .attr("y", height - 40)
  .attr("text-anchor", "start")
  .attr("fill", "gray")
  .attr("font-size", "1em")
  .text("X-Perp");

svg
  .append("text")
  .attr("x", height / 2)
  .attr("y", 27 - width)
  .attr("transform", "rotate(90)")
  .attr("text-anchor", "end")
  .attr("fill", "gray")
  .attr("font-size", "1em")
  .text("C-Perp");

d3.csv(
  data_server + "/perplexity_tiny",
  ({ id, x_perp, c_perp, label}) => ({
    img_id: id,
    x: +x_perp,
    y: +c_perp,
    label: label,
    //top_x: x_top,
    //top_c: c_top,
  })
).then(draw_canvas);

var in_range = (x, range) => d3.min(range) <= x && x <= d3.max(range);

function draw_canvas(data) {
  var x_tf = d3
    .scaleLinear()
    .domain(d3.extent(data, (d) => d.x))
    .range([margin.left + 2 * radius, width - margin.right - 2 * radius]);
  var x_range = x_tf.domain();

  var y_tf = d3
    .scaleLinear()
    .domain(d3.extent(data, (d) => d.y))
    .range([height - margin.bottom - 2 * radius, margin.top + 2 * radius]);
  var y_range = y_tf.domain();

  var x_axis_ref = svg.append("g").call(xAxis, x_tf);
  var grid_x_ref = svg.append("g").call(x_grid, x_tf);

  var y_axis_ref = svg.append("g").call(yAxis, y_tf);
  var grid_y_ref = svg.append("g").call(y_grid, y_tf);

  var hexbin = d3
    .hexbin()
    .x((d) => x_tf(d.x))
    .y((d) => y_tf(d.y))
    .radius(radius)
    .extent([
      [margin.left, margin.top],
      [width - margin.right, height - margin.bottom],
    ]);

  var data_s = data.slice(0, Math.floor(data.length / 5));
  var bins_f = hexbin(data_s);
  //var bins_f = bins.filter(d => in_range(d.x, [margin.left, width - margin.right]) &&
  //        in_range(d.y, [margin.left, height - margin.bottom]));
  var color = d3
    .scaleSequentialSymlog(d3.interpolateBuPu)
    .domain([0, 2 * d3.max(bins_f, (d) => d.length)]);
  /* 
    r = d3.scaleSymlog()
        .domain([0, d3.max(bins, d => d.length)])
        .range([0, hexbin.radius()])
    */

  var data_hex_ref = svg
    .append("g")
    .attr("stroke", "#000")
    .attr("stroke-opacity", 0);

  data_hex_ref
    .selectAll("path")
    .data(bins_f)
    .join("path")
    .call(draw_hexagon, hexbin, color);

  var data_hex_select_ref = svg
    .append("g")
    .attr("stroke", "#000")
    .attr("stroke-opacity", 0);

  var selected_class = null;
  var selected = false;
  var selected_color = d3.scaleSequentialSymlog(d3.interpolatePuBu);

  var data_class_group = d3.group(data, (d) => d.label);
  var data_class = Array.from(data_class_group).map((d) => ({
    label: d[0],
    x: d3.mean(d[1], (d) => d.x),
    y: d3.mean(d[1], (d) => d.y),
  }));

  var class_marker_bak = svg
    .append("g")
    .attr("opacity", 0.05)
    .attr("stroke", "#0001");

  var class_marker_ref = svg
    .append("g")
    .attr("opacity", 0.05)
    .attr("stroke", "#0001");

  class_marker_ref
    .selectAll("path")
    .data(data_class)
    .join("path")
    .attr(
      "transform",
      (d) => `translate(${x_tf(d.x)}, ${y_tf(d.y)}) rotate(45)`
    )
    .attr(
      "d",
      d3
        .symbol()
        .type(d3.symbolCross)
        .size(15 * radius)
    )
    .attr("fill", "#fc7")
    .on("mouseover", function () {
      d3.select(this).call(class_on);
    })
    .on("click", class_select)
    .on("mouseout", function () {
      d3.select(this).call(class_out);
    });

  var data_x = data_s.map((d) => d.x);
  var data_y = data_s.map((d) => d.y);

  var x_bins = d3.histogram().domain(x_tf.domain()).thresholds(x_tf.ticks(150))(
    data_x
  );
  var x_bins_max = Math.max(
    1,
    d3.max(x_bins, (d) => d.length)
  );

  var y_bins = d3.histogram().domain(y_tf.domain()).thresholds(y_tf.ticks(150))(
    data_y
  );
  var y_bins_max = Math.max(
    1,
    d3.max(y_bins, (d) => d.length)
  );

  var x_bar = svg.append("g").attr("fill", "steelblue");

  x_bar
    .selectAll("rect")
    .data(x_bins)
    .join("rect")
    .call(draw_x_bins, x_tf, x_bins_max);

  var y_bar = svg.append("g").attr("fill", "steelblue");

  y_bar
    .selectAll("rect")
    .data(y_bins)
    .join("rect")
    .call(draw_y_bins, y_tf, y_bins_max);

  var zoom = d3
    .zoom()
    .scaleExtent([0, Infinity])
    .extent([
      [margin.left, margin.top],
      [width - margin.right, height - margin.bottom],
    ])
    .translateExtent([
      [margin.left, margin.top],
      [width - margin.right, height - margin.bottom],
    ])
    .on("start", rec_pos)
    .on("zoom", zoomed);

  var start_pos = [0, 0];

  function rec_pos() {
    var cord = d3.mouse(this);
    start_pos = [x_tf.invert(cord[0]), y_tf.invert(cord[1])];
  }

  svg.call(zoom);
  var kx = 1.0,
    ky = 1.0,
    prev_k = 1.0;

  function zoomed() {
    var cord = d3.mouse(this);
    var px = cord[0],
      py = cord[1];

    var k = d3.event.transform.k;
    var x_center = x_tf.invert(px),
      y_center = y_tf.invert(py);

    if (px < width - margin.right) {
      kx = (kx * k) / prev_k;
      kx = Math.max(1, kx);
      var domain = x_tf.domain();
      var x_range_k = (x_range[1] - x_range[0]) / kx;
      var rs_fac = (x_range[1] - x_range[0]) / (domain[1] - domain[0]);
      domain = domain.map((x) => (rs_fac * (x - x_center)) / kx + start_pos[0]);
      domain =
        domain[1] - x_range_k < x_range[0]
          ? [x_range[0], x_range[0] + x_range_k]
          : domain;
      domain =
        domain[0] + x_range_k > x_range[1]
          ? [x_range[1] - x_range_k, x_range[1]]
          : domain;
      x_tf.domain(domain);
    }

    if (py < height - margin.bottom) {
      ky = (ky * k) / prev_k;
      ky = Math.max(1, ky);
      var domain = y_tf.domain();
      var y_range_k = (y_range[1] - y_range[0]) / ky;
      var rs_fac = (y_range[1] - y_range[0]) / (domain[1] - domain[0]);
      domain = domain.map((y) => (rs_fac * (y - y_center)) / ky + start_pos[1]);
      domain =
        domain[1] - y_range_k <= y_range[0]
          ? [y_range[0], y_range[0] + y_range_k]
          : domain;
      domain =
        domain[0] + y_range_k >= y_range[1]
          ? [y_range[1] - y_range_k, y_range[1]]
          : domain;
      y_tf.domain(domain);
    }
    prev_k = k;

    //filter bins not in the svg range
    var data_f = data
      .slice(0, Math.min(data.length, Math.floor((kx * ky * data.length) / 5)))
      .filter(
        (d) => in_range(d.x, x_tf.domain()) && in_range(d.y, y_tf.domain())
      );
    bins_f = hexbin(data_f);

    x_bins = d3.histogram().domain(x_tf.domain()).thresholds(x_tf.ticks(200))(
      data_f.map((d) => d.x)
    );
    y_bins = d3.histogram().domain(y_tf.domain()).thresholds(y_tf.ticks(150))(
      data_f.map((d) => d.y)
    );

    x_bar.selectAll("rect").remove();
    x_bar
      .selectAll("rect")
      .data(x_bins)
      .join("rect")
      .call(
        draw_x_bins,
        x_tf,
        Math.max(
          1,
          d3.max(x_bins, (d) => d.length)
        )
      );

    y_bar.selectAll("rect").remove();
    y_bar
      .selectAll("rect")
      .data(y_bins)
      .join("rect")
      .call(
        draw_y_bins,
        y_tf,
        Math.max(
          1,
          d3.max(y_bins, (d) => d.length)
        )
      );

    color.domain([0, 2 * d3.max(bins_f, (d) => d.length)]);

    x_axis_ref.call(xAxis, x_tf);
    grid_x_ref.call(x_grid, x_tf);

    y_axis_ref.call(yAxis, y_tf);
    grid_y_ref.call(y_grid, y_tf);

    data_hex_ref.selectAll("path").remove();

    data_hex_ref
      .selectAll("path")
      .data(bins_f)
      .join("path")
      .call(draw_hexagon, hexbin, color);

    if (selected_class) {
      var selected_bins = hexbin(
        data_class_group
          .get(selected_class, [])
          .filter(
            (d) => in_range(d.x, x_tf.domain()) && in_range(d.y, y_tf.domain())
          )
      );

      data_hex_select_ref.selectAll("path").remove();

      data_hex_select_ref
        .selectAll("path")
        .data(selected_bins)
        .join("path")
        .call(draw_hexagon, hexbin, selected_color);
    }

    class_marker_bak
      .attr("opacity", Math.min(1, 0.05 * (kx * ky)))
      .selectAll("path")
      .attr(
        "transform",
        (d) => `translate(${x_tf(d.x)}, ${y_tf(d.y)}) rotate(45)`
      ) //
      .attr("visibility", (d) =>
        !selected &&
        in_range(d.x, x_tf.domain()) &&
        in_range(d.y, y_tf.domain())
          ? "visible"
          : "hidden"
      );

    class_marker_ref
      .attr("opacity", selected || selected_class ? 1 : Math.min(1, 0.05 * (kx * ky)))
      .selectAll("path")
      .attr(
        "transform",
        (d) =>
          `translate(${x_tf(d.x)}, ${y_tf(d.y)}) rotate(${selected || selected_class ? 90 : 45})`
      ) //
      .attr("visibility", (d) =>
        in_range(d.x, x_tf.domain()) && in_range(d.y, y_tf.domain())
          ? "visible"
          : "hidden"
      );
  }

  function class_on(class_point) {
    if (!selected) {
      class_marker_ref
        .selectAll("path")
        .filter((d) => d.label != class_point.data()[0].label)
        .remove()
        .nodes()
        .forEach((d) => class_marker_bak.append(() => d));

      class_marker_bak
        .selectAll("path")
        .filter((d) => d.label == class_point.data()[0].label)
        .remove()
        .nodes()
        .forEach((d) => class_marker_ref.append(() => d));

      class_marker_ref.attr("opacity", 1);

      class_marker_bak.transition().delay(500).duration(150).attr("opacity", 0);

      class_point
        .attr(
          "d",
          d3
            .symbol()
            .type(d3.symbolCross)
            .size(50 * radius)
        )
        .transition()
        .delay(500)
        .duration(150)
        .attr(
          "transform",
          (d) => `translate(${x_tf(d.x)}, ${y_tf(d.y)}) rotate(90)`
        );

      data_hex_ref.transition().delay(500).duration(250).attr("opacity", 0.2);
      x_bar.transition().delay(500).duration(250).attr("opacity", 0.2);
      y_bar.transition().delay(500).duration(250).attr("opacity", 0.2);


      selected_class = class_point.data()[0].label;
      var selected_bins = hexbin(
        data_class_group
          .get(selected_class)
          .filter(
            (d) => in_range(d.x, x_tf.domain()) && in_range(d.y, y_tf.domain())
          )
      );
      data_hex_select_ref
        .attr("opacity", 0)
        .selectAll("path")
        .data(selected_bins)
        .join("path")
        .call(
          draw_hexagon,
          hexbin,
          selected_color.domain([0, 2 * d3.max(selected_bins, (d) => d.length)])
        );

      data_hex_select_ref
        .transition()
        .delay(500)
        .duration(250)
        .attr("opacity", 1);
    } else {
      selected_class = null;
      class_point
        .transition()
        .duration(150)
        .attr(
          "transform",
          (d) => `translate(${x_tf(d.x)}, ${y_tf(d.y)}) rotate(45)`
        );

      data_hex_ref.transition().duration(250).attr("opacity", 1);
      x_bar.transition().duration(250).attr("opacity", 1);
      y_bar.transition().duration(250).attr("opacity", 1);

      data_hex_select_ref.transition().duration(250).attr("opacity", 0);
    }
  }

  function class_select() {
    selected = !selected;
    class_marker_bak
      .attr("opacity", Math.min(1, 0.05 * (kx * ky)))
      .selectAll("path")
      .transition()
      .duration(100)
      .attr("visibility", (d) =>
        !selected &&
        in_range(d.x, x_tf.domain()) &&
        in_range(d.y, y_tf.domain())
          ? "visible"
          : "hidden"
      );
    d3.select("#image_info_page").style("display", selected ? "none" : "block");
    d3.select("#class_info_page").style(
      "display",
      !selected ? "none" : "block"
    );

    var point = d3.select(this).data()[0];

    d3.select("#class_selection").attr(
      "src",
      `images/ILSVRC2012_val_${("00000000" + repr[point.label]).slice(
        -8
      )}.JPEG`
    );
    d3.select("#class_label").text(point.label);
    var info_box = d3.select("#class_info");
    info_box
      .select("#class_name")
      .text(label2name[point.label])
      .on("mouseover", function (d) {
        tooltip
          .html(gen_intro(point.label))
          .style("top", d3.event.pageY - 10 + "px")
          .style("left", d3.event.pageX + 10 + "px")
          .transition()
          .delay(500)
          .duration(250)
          .style("opacity", 1);
        over_class(point.label);
      })
      .on("mouseout", function () {
        tooltip.transition().style("opacity", 0);
        out_class(point.label);
      })
      .on("click", function () {
        select_class(point.label);
      });

    info_box
      .select("#class_x_perplexity")
      .text((100 * point.x).toFixed(1) + "% avg");
    info_box.select("#class_c_perplexity").text(point.y.toFixed(4) + " avg");
    var get_dict = (d) =>
      d.reduce(function (dict, pair) {
        dict[pair[0]] = pair[1];
        return dict;
      }, {});
    d3.json(data_server + `/details/c/${point.label}`).then(
        function(data) {
            point.top_v = data["voting"];
            point.top_e = data["expectation"];
            var v_pie_data = get_dict(point.top_v);
            v_pie_data.others = 1 - Object.values(v_pie_data).reduce((a, b) => a + b);
            v_pie_data = class_v_pie(d3.entries(v_pie_data));
            class_v_pie_info
            .selectAll("path")
            .data(v_pie_data)
            .join("path")
            .transition()
            .duration(400)
            .attrTween("d", arcTween)
            .attr("fill", (d) =>
                d.index == v_pie_data.length - 1 ? "#0000" : pie_color_class(d.index)
            )
            .attr("opacity", 0.7)
            .attr("stroke", "gray")
            .attr("stroke-opacity", 0.1);

            v_pie_data.sort((a, b) => a.index - b.index);
            d3.select("#class_v_top_classes_list")
            .selectAll("li")
            .data(v_pie_data)
            .join("li")
            .text(
                (d) =>
                label2name[d.data.key] + ": " + (100 * d.data.value).toFixed(2) + "%"
            )
            .filter((d) => d.data.key != "others")
            .on("mouseover", function (d) {
                tooltip
                .html(gen_intro(d.data.key))
                .style("top", d3.event.pageY - 10 + "px")
                .style("left", d3.event.pageX + 10 + "px")
                .transition()
                .delay(500)
                .duration(250)
                .style("opacity", 1);
                over_class(d.data.key);
            })
            .on("mouseout", function (d) {
                tooltip.transition().style("opacity", 0);
                out_class(d.data.key);
            })
            .on("click", function (d) {
                select_class(d.data.key);
            });

            var e_pie_data = get_dict(point.top_e);
            e_pie_data.others = 1 - Object.values(e_pie_data).reduce((a, b) => a + b);
            e_pie_data = class_e_pie(d3.entries(e_pie_data));
            class_e_pie_info
            .selectAll("path")
            .data(e_pie_data)
            .join("path")
            .transition()
            .duration(100)
            .attrTween("d", arcTween)
            .attr("fill", (d) =>
                d.index == e_pie_data.length - 1 ? "#0000" : pie_color_class(d.index)
            )
            .attr("opacity", 0.7)
            .attr("stroke", "gray")
            .attr("stroke-opacity", 0.1);

            e_pie_data.sort((a, b) => a.index - b.index);
            d3.select("#class_e_top_classes_list")
            .selectAll("li")
            .data(e_pie_data)
            .join("li")
            .text(
                (d) =>
                label2name[d.data.key] + ": " + (100 * d.data.value).toFixed(2) + "%"
            )
            .filter((d) => d.data.key != "others")
            .on("mouseover", function (d) {
                tooltip
                .html(gen_intro(d.data.key))
                .style("top", d3.event.pageY - 10 + "px")
                .style("left", d3.event.pageX + 10 + "px")
                .transition()
                .delay(500)
                .duration(250)
                .style("opacity", 1);
                over_class(d.data.key);
            })
            .on("mouseout", function (d) {
                tooltip.transition().style("opacity", 0);
                out_class(d.data.key);
            })
            .on("click", function (d) {
                select_class(d.data.key);
            });

            //dendro_class(point.label);
        }
    )
  }

  function class_out(class_point) {
    if (selected) {
      selected_class = class_point.data()[0].label;

      class_point
        .transition()
        .delay(500)
        .duration(150)
        .attr(
          "transform",
          (d) => `translate(${x_tf(d.x)}, ${y_tf(d.y)}) rotate(90)`
        );

      data_hex_ref.transition().delay(500).duration(250).attr("opacity", 0.2);
      x_bar.transition().delay(500).duration(250).attr("opacity", 0.2);
      y_bar.transition().delay(500).duration(250).attr("opacity", 0.2);

      data_hex_select_ref
        .transition()
        .delay(500)
        .duration(250)
        .attr("opacity", 1);
    } else {
      class_point
        .transition()
        .duration(150)
        .attr(
          "d",
          d3
            .symbol()
            .type(d3.symbolCross)
            .size(15 * radius)
        )
        .attr(
          "transform",
          (d) => `translate(${x_tf(d.x)}, ${y_tf(d.y)}) rotate(45)`
        );

      class_marker_ref.attr("opacity", Math.min(1, 0.05 * (kx * ky)));

      class_marker_bak
        .transition()
        .delay(500)
        .duration(150)
        .attr("opacity", Math.min(1, 0.05 * (kx * ky)));

      class_marker_bak
        .selectAll("path")
        .remove()
        .nodes()
        .forEach((d) => class_marker_ref.append(() => d));

      data_hex_ref.transition().duration(250).attr("opacity", 1);
      x_bar.transition().duration(250).attr("opacity", 1);
      y_bar.transition().duration(250).attr("opacity", 1);

      data_hex_select_ref.transition().duration(250).attr("opacity", 0);

      data_hex_select_ref.selectAll("path").remove();
      selected_class = null;
    }
  }

  over_class = (label) =>
    class_marker_ref
      .selectAll("path")
      .filter((d) => d.label == label)
      .call((d) => (d.data().length ? class_on(d) : null));

  out_class = (label) =>
    class_marker_ref
      .selectAll("path")
      .filter((d) => d.label == label)
      .call((d) => (d.data().length ? class_out(d) : null));

  select_class = (label) =>
    class_marker_ref
      .selectAll("path")
      .filter((d) => d.label == label)
      .call((d) => (d.data().length ? class_select(d) : null));
}

var pie_radius = 40;
var pie_box_len = 10 + pie_radius * 2;
var pie_offset = pie_box_len / 2;
var img_v_pie_info = d3
  .select("#image_v_top_classes")
  .attr("height", pie_box_len)
  .attr("width", pie_box_len)
  .append("g")
  .attr("transform", "translate(" + pie_offset + ", " + pie_offset + ")");

var img_e_pie_info = d3
  .select("#image_e_top_classes")
  .attr("height", pie_box_len)
  .attr("width", pie_box_len)
  .append("g")
  .attr("transform", "translate(" + pie_offset + ", " + pie_offset + ")");

var class_v_pie_info = d3
  .select("#class_v_top_classes")
  .attr("height", pie_box_len)
  .attr("width", pie_box_len)
  .append("g")
  .attr("transform", "translate(" + pie_offset + ", " + pie_offset + ")");

var class_e_pie_info = d3
  .select("#class_e_top_classes")
  .attr("height", pie_box_len)
  .attr("width", pie_box_len)
  .append("g")
  .attr("transform", "translate(" + pie_offset + ", " + pie_offset + ")");

var pie_color = d3.scaleSequential(d3.interpolatePuBu).domain([5, 0]);
var pie_color_class = d3.scaleSequential(d3.interpolateOranges).domain([5, 0]);
var arc = d3
  .arc()
  .innerRadius(pie_radius / 2)
  .outerRadius(pie_radius);

function arcTween(d) {
  var inter = d3.interpolate(this._current, d);
  this._current = inter(0);
  return function (t) {
    _current = inter(t);
    return arc(_current);
  };
}

var img_v_pie = d3
  .pie()
  .value((d) => d.value)
  .sort(null);
var img_e_pie = d3
  .pie()
  .value((d) => d.value)
  .sort(null);

var class_v_pie = d3
  .pie()
  .value((d) => d.value)
  .sort(null);
var class_e_pie = d3
  .pie()
  .value((d) => d.value)
  .sort(null);

var xAxis = (g, x_tf) =>
  g
    .attr("transform", `translate(0, ${height - margin.bottom - 5 + radius})`)
    .call(d3.axisBottom(x_tf).ticks(width / 80, ""))
    .attr("stroke", "gray")
    .attr("opacity", 0.5)
    .call((g) => g.select(".domain").remove());

var yAxis = (g, y_tf) =>
  g
    .attr("transform", `translate(${width - margin.right - 5 + radius},0)`)
    .call(d3.axisRight(y_tf).ticks(height / 50, ""))
    .attr("stroke", "gray")
    .attr("opacity", 0.5)
    .call((g) => g.select(".domain").remove());

var x_grid = (g, x_tf) =>
  g
    .attr("transform", `translate(0, ${height - margin.bottom})`)
    .attr("stroke", "gray")
    .attr("opacity", 0.1)
    .call(
      d3
        .axisBottom(x_tf)
        .tickSize(-height + margin.top + margin.bottom)
        .tickFormat("")
        .ticks(10)
    );

var y_grid = (g, y_tf) =>
  g
    .attr("transform", `translate(${margin.left}, 0)`)
    .attr("stroke", "gray")
    .attr("opacity", 0.1)
    .call(
      d3
        .axisLeft(y_tf)
        .tickSize(-width + margin.left + margin.right)
        .tickFormat("")
        .ticks(10)
    );

var draw_x_bins = (g, x_tf, x_bins_max) =>
  g
    .attr("opacity", 0.5)
    .attr("x", (d) => x_tf(d.x1))
    .attr("width", width / 150)
    .attr("y", (d) => height - margin.bottom + 25)
    .attr("height", (d) => (d.length / x_bins_max) * 40);

var draw_y_bins = (g, y_tf, y_bins_max) =>
  g
    .attr("opacity", 0.5)
    .attr("x", (d) => width - margin.right + 40)
    .attr("width", (d) => (d.length / y_bins_max) * 40)
    .attr("y", (d) => y_tf(d.x1))
    .attr("height", height / 150);

var tooltip = d3
  .select("body")
  .append("div")
  .attr("class", "tooltip")
  .style("opacity", 0)
  .style("position", "absolute");

function click(d) {
  d3.select("#image_info_page").style("display", "block");
  d3.select("#class_info_page").style("display", "none");
  var point = d[Math.floor(Math.random() * d.length)];

  var image_name = `ILSVRC2012_val_${("00000000" + point.img_id).slice(
    -8
  )}.JPEG`;
  d3.select("#selection").attr("src", "images/" + image_name);
  d3.select("#image_name").text(image_name);
  var info_box = d3.select("#info");
  info_box
    .select("#image_label")
    .text(label2name[point.label])
    .on("mouseover", function (d) {
      tooltip
        .html(gen_intro(point.label))
        .style("top", d3.event.pageY - 10 + "px")
        .style("left", d3.event.pageX + 10 + "px")
        .transition()
        .delay(500)
        .duration(250)
        .style("opacity", 1);
      over_class(point.label);
    })
    .on("mouseout", function () {
      tooltip.transition().style("opacity", 0);
      out_class(point.label);
    })
    .on("click", function () {
      select_class(point.label);
    });

  info_box.select("#image_x_perplexity").text((100 * point.x).toFixed(1) + "%");
  info_box.select("#image_c_perplexity").text(point.y.toFixed(4));
  var get_dict = (d) =>
    JSON.parse(d).reduce(function (dict, pair) {
      dict[pair[0]] = pair[1];
      return dict;
    }, {});
  d3.json(data_server+`/details/i/${point.img_id}`).then(
      function(data) {
          point.top_v = data["v_top"];
          point.top_e = data["e_top"];
          var v_pie_data = get_dict(point.top_v);
          v_pie_data.others = 500 - Object.values(v_pie_data).reduce((a, b) => a + b);
          for (key in v_pie_data) {
              v_pie_data[key] = v_pie_data[key] / 500;
          }
          v_pie_data = img_v_pie(d3.entries(v_pie_data));
          img_v_pie_info
              .selectAll("path")
              .data(v_pie_data)
              .join("path")
              .transition()
              .duration(400)
              .attrTween("d", arcTween)
              .attr("fill", (d) =>
              d.index == v_pie_data.length - 1 ? "#0000" : pie_color(d.index)
              )
              .attr("opacity", 0.7)
              .attr("stroke", "gray")
              .attr("stroke-opacity", 0.1);

          v_pie_data.sort((a, b) => a.index - b.index);
          d3.select("#image_v_top_classes_list")
              .selectAll("li")
              .data(v_pie_data)
              .join("li")
              .text(
              (d) =>
                  label2name[d.data.key] + ": " + (100 * d.data.value).toFixed(2) + "%"
              )
              .filter((d) => d.data.key != "others")
              .on("mouseover", function (d) {
              tooltip
                  .html(gen_intro(d.data.key))
                  .style("top", d3.event.pageY - 10 + "px")
                  .style("left", d3.event.pageX + 10 + "px")
                  .transition()
                  .delay(500)
                  .duration(250)
                  .style("opacity", 1);
              over_class(d.data.key);
              })
              .on("mouseout", function (d) {
              tooltip.transition().style("opacity", 0);
              out_class(d.data.key);
              })
              .on("click", function (d) {
              select_class(d.data.key);
              });

          var e_pie_data = get_dict(point.top_e);
          e_pie_data.others =
              1 - Object.values(e_pie_data).reduce((a, b) => a + b);
          e_pie_data = img_e_pie(d3.entries(e_pie_data));
          img_e_pie_info
              .selectAll("path")
              .data(e_pie_data)
              .join("path")
              .transition()
              .duration(100)
              .attrTween("d", arcTween)
              .attr("fill", (d) =>
              d.index == e_pie_data.length - 1 ? "#0000" : pie_color(d.index)
              )
              .attr("opacity", 0.7)
              .attr("stroke", "gray")
              .attr("stroke-opacity", 0.1);

          e_pie_data.sort((a, b) => a.index - b.index);
          d3.select("#image_e_top_classes_list")
              .selectAll("li")
              .data(e_pie_data)
              .join("li")
              .text(
              (d) =>
                  label2name[d.data.key] + ": " + (100 * d.data.value).toFixed(2) + "%"
              )
              .filter((d) => d.data.key != "others")
              .on("mouseover", function (d) {
              tooltip
                  .html(gen_intro(d.data.key))
                  .style("top", d3.event.pageY - 10 + "px")
                  .style("left", d3.event.pageX + 10 + "px")
                  .transition()
                  .delay(500)
                  .duration(250)
                  .style("opacity", 1);
              over_class(d.data.key);
              })
              .on("mouseout", function (d) {
              tooltip.transition().style("opacity", 0);
              out_class(d.data.key);
              })
              .on("click", function (d) {
              select_class(d.data.key);
              });

      }
  )
  //dendro_class(point.label);
}

function gen_intro(label) {
  var img_content =
    label == "others"
      ? " "
      : "<img src=" +
        `"images/ILSVRC2012_val_${("00000000" + repr[label]).slice(
          -8
        )}.JPEG"></img>`;
  return (
    "" +
    "<div><span>" +
    label +
    "</span></div>" +
    img_content +
    "<p>" +
    label2desc[label] +
    "</p>"
  );
}

var draw_hexagon = (path, hexbin, color) =>
  path
    .attr("d", (d) => hexbin.hexagon())
    .attr("transform", (d) => `translate(${d.x}, ${d.y})`)
    .attr("fill", (d) => color(d.length))
    .on("mouseover", function () {
      d3.select(this).attr("fill", "#AAAA");
    })
    .on("mouseout", function () {
      d3.select(this).attr("fill", (d) => color(d.length));
    })
    .on("click", click);
