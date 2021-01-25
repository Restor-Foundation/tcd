import React from 'react';
import ReactDOM from 'react-dom';
import mapboxgl from 'mapbox-gl';
import MapboxDraw from "@mapbox/mapbox-gl-draw";
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';
import { CircleMode, DragCircleMode, DirectMode, SimpleSelectMode } from 'mapbox-gl-draw-circle';
import * as turf from "@turf/turf";

/*global fetch*/
//constants
mapboxgl.accessToken = 'pk.eyJ1IjoiYmxpc2h0ZW4iLCJhIjoiMEZrNzFqRSJ9.0QBRA2HxTb8YHErUFRMPZg';
const PYTHON_REST_SERVER_ENDPOINT = 'https://andrewcottam.com:8081/python-rest-server/restor/services/';
const GEE_IMAGE_SERVER_ENDPOINT = 'https://geeimageserver.appspot.com/';
var draw, map;
class Application extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      lng: -69.44916228299296,
      lat: -12.833871131544694,
      zoom: 17
    };
  }

  componentDidMount() {
    map = new mapboxgl.Map({
      container: this.mapContainer,
      style: 'mapbox://styles/mapbox/streets-v11',
      center: [this.state.lng, this.state.lat],
      zoom: this.state.zoom
    });
    map.on('move', () => {
      this.setState({
        lng: map.getCenter().lng.toFixed(4),
        lat: map.getCenter().lat.toFixed(4),
        zoom: map.getZoom().toFixed(2)
      });
    });
    map.on('load', () => {
      let _center = turf.point([0, 40]);
      let _radius = 25;
      let _options = {
        steps: 80,
        units: 'kilometers' // or "mile"
      };
      let _circle = turf.circle(_center, _radius, _options);
      map.addSource("circleData", {
        type: "geojson",
        data: _circle,
      });
      map.addLayer({
        id: "circle-fill",
        type: "fill",
        source: "circleData",
        paint: {
          "fill-color": "red",
          "fill-opacity": 1,
        },
      });
      //create the draw control
      draw = new MapboxDraw({
        defaultMode: "draw_circle",
        displayControlsDefault: false,
        userProperties: true,
        modes: {
          ...MapboxDraw.modes,
          draw_circle: CircleMode,
          drag_circle: DragCircleMode,
          direct_select: DirectMode,
          simple_select: SimpleSelectMode
        }
      });
      map.addControl(draw);
      draw.changeMode('drag_circle');
      map.on('draw.selectionchange', (e) => {
        draw.changeMode('drag_circle');
      });
      //move to the first image
      this.nextImage();
    });
  }
  _get(url) {
    return new Promise((resolve, reject) => {
      fetch(url).then(response => {
        response.json().then(_json => {
          resolve(_json);
        });
      });
    });
  }

  //writes the records to postgis
  postFeatures() {
    const features = draw && draw.getAll().features;
    if (features && features.length > 0) {
      features.forEach(feature => {
        if (feature.properties.center.length > 0) {
          const center = feature.properties.center;
          this._get(PYTHON_REST_SERVER_ENDPOINT + "set_tcd_feature?_id=" + feature.id + "&_longitude=" + center[0] + "&_latitude=" + center[1] + "&_radius=" + (feature.properties.radiusInKm * 1000) + "&_gee_imageid=" + "gee1234" + "&_entered_by=" + "andrew");
        }
      });
    }
  }
  //gets the next image using the current states lat/long
  nextImage() {
    this.postFeatures();
    //get the scenes for the next lat/lng
    const collectionid = 'GOOGLE/GEO/ALL/SATELLITE/WORLDVIEW3/ORTHO/RGB';
    // const collectionid = 'LANDSAT/LC8_L1T_TOA';
    this._get(GEE_IMAGE_SERVER_ENDPOINT + "getIdsForPoint?lng=" + this.state.lng + "&lat=" + this.state.lat + "&collectionid=" + collectionid).then(json => {
      //the geeImageServer returns quasi-json data
      const sceneIds = eval(json.records.substring(json.records.indexOf("["), json.records.length - 1));
      //if there are scenes then get the first
      const sceneId = (sceneIds.length) ? sceneIds[0] : undefined;
      //load that scene
      if (sceneId) this.addGEEImage(sceneId);
    });
  }
  //loads a GEE scene from the geeImageServer
  addGEEImage(sceneId) {
    map.addSource('gee-source', {
      'type': 'raster',
      'tiles': [
        'https://geeimageserver.appspot.com/ogc?service=WMS&request=GetMap&version=1.1.1&styles=&format=image%2Fpng&transparent=false&layers=[' + sceneId + ']&srs=EPSG%3A3857&bbox={bbox-epsg-3857}'
      ],
      'tileSize': 256
    });
    map.addLayer({
        'id': 'gee-layer',
        'type': 'raster',
        'source': 'gee-source',
        'paint': {}
      },
      "gl-draw-polygon-fill-inactive.cold");
  }

  render() {
    return (
      <div>
        <div className='sidebarStyle'>
          <div>Longitude: {this.state.lng} | Latitude: {this.state.lat} | Zoom: {this.state.zoom}</div>
        </div>
        <div ref={el => this.mapContainer = el} className='mapContainer' />
        <div onClick={this.nextImage.bind(this)} className={'next'}>Next</div>
      </div>
    );
  }
}

ReactDOM.render(<Application />, document.getElementById('app'));
