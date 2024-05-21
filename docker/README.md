# Dockerfiles

Usage:

```bash
build.sh Dockerfile
```

or on an ARM machine, e.g. a Mac:

```bash
build.sh Dockerfile_arm
```

## Running

```bash
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix tcd 
```

This will place you inside the container in the repository folder as the `restor` user. You can run the test suite to make sure that everything is running as expected:

```bash
python -m pytest
```