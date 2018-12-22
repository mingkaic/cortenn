FROM mkaichen/bazel_cpp:latest

ENV APP_DIR /usr/src/tenncor

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

COPY . $APP_DIR

CMD [ "./tests.sh" ]
