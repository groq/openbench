export DOCKER_BUILDKIT=0
docker build -t sekai_web_waffrog:latest .
docker run -d -p 1337:1337 --rm --name=sekai_web_waffrog sekai_web_waffrog:latest