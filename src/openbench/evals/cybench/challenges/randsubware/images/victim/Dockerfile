FROM python:3.9-slim-buster

RUN apt-get update -y && \
   apt-get install -y lib32z1 xinetd socat && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/*

RUN useradd -m user && \
   chown -R root:root /home/user

COPY app /home/user/
COPY xinetd /etc/xinetd.d/user

WORKDIR /home/user

EXPOSE 9999

CMD ["/usr/sbin/xinetd", "-dontfork"]

# Set the working directory in the container
#WORKDIR /app

# Copy the necessary files
#COPY app/chall.py /app/chall.py
#COPY app/flag.py /app/flag.py

# Set file permissions
#RUN chmod 755 /app/chall.py
#RUN chmod 700 /app/flag.py

# Add a new user "appuser" with no password, set home
#RUN adduser --disabled-password --gecos '' --home /app --shell /bin/bash appuser

## Change the owner of the app directory to appuser
#RUN chown -R appuser:appuser /app
#RUN chown root:root /app/flag.py  # Keep flag.py owned by root

# Change to non-root privilege
#USER appuser

# Open a shell at start-up
#CMD ["/bin/bash"]
