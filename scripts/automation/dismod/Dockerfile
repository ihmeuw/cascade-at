FROM debian:stretch

RUN apt-get update \ 
	&& apt-get install -y --no-install-recommends \
	    apt-utils \
		build-essential \
		gfortran \
		autoconf \
		automake \
		git-all \
		wget \
		sed \
		tar \
		cmake \
		libatlas-base-dev \
		libghc-hmatrix-gsl-dev \
	    libgsl-dev \
	    liblapack-dev \
		libsqlite3-0 \
		libsqlite3-dev \
		pkg-config \
		python3 \
		python3-dev \
		python3-mysqldb \
		python3-numpy \
		python3-pandas \
		python3-pip \
		python3-scipy \
		python3-setuptools \
		python3-sqlalchemy \
		sqlite3 \
		libsuitesparse-dev \
	&& pip3 install pymysql pysqlite3

#RUN apt-get update locate \
#    && updatedb

ENV HOME="/home/root"
ENV DMROOT="${HOME}/prefix/dismod_at.release"
ENV PYTHONPATH="${DMROOT}/lib/python3.5/site-packages"
ENV PATH="${PATH}:${DMROOT}/bin"
ENV LD_LIBRARY_PATH="${DMROOT}/lib64"
#ENV LD_LIBRARY_PATH="${DMROOT}/release/lib64"
ENV PKG_CONFIG_PATH="${DMROOT}/lib64/pkgconfig"

WORKDIR /usr/src/app

ADD dismod_at.tgz ./
WORKDIR /usr/src/app/dismod_at

RUN mkdir -p "$HOME/prefix" \
	&& sed -i '40,60s/debug/release/' bin/run_cmake.sh \
    && bin/install_eigen.sh \
    && bin/install_ipopt.sh \
    && bin/install_cppad.sh \
    && bin/install_cppad_mixed.sh \
    && bin/run_cmake.sh \
	&& cd build \
    && make speed \
    && make install

# This is for the IHME cluster.
RUN ln -s /snfs1 /home/j
# Set timezone to PST
ENV TZ="PST"
RUN rm /etc/localtime
ENV TZ=America/Los_Angeles
RUN ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime


ENV PATH="${PATH}:/usr/local/UGE/bin/lx-amd64"

# Removed IHME mounts here. Can add them dynamically.

RUN chmod 777 /etc /var

CMD ["/home/root/prefix/dismod_at/bin/dismod_at"]
