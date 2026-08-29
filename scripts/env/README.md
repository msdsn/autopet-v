# Environment glue

Helper scripts for the GPU machines this work was developed on: code sync, runtime bootstrap,
training launch/resume and a training-progress watcher. They hard-code that environment's paths
and are not needed to build the container or to reproduce an evaluation run.
