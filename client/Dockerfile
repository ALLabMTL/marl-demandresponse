FROM alpine:latest
RUN apk add nodejs npm python3 python3 py3-pip --no-cache
COPY ./package.json /src/package.json
WORKDIR /src/
RUN npm install
COPY . /src/
RUN npm run build
CMD ["python3","-m","http.server","--bind","0.0.0.0","--directory","./dist/prj/","4200"]
