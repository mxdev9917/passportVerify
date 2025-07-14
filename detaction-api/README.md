python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


docker buildx -t detector .
docker build --no-cache -t detector .
docker tag detector:latest mx9917/detector:latest

docker run -it --rm -p 5000:5000 passport-detector



docker build -t mx9917/detector .
docker buildx create --use
docker buildx build --platform linux/amd64 -t mx9917/detector:latest . --push

docker push mx9917/detector:latest