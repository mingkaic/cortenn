FROM mkaichen/bazel_cpp_tf:latest

ENV APP_DIR /usr/src/cortenn

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

COPY . $APP_DIR

CMD [ "./tests.sh" ]
