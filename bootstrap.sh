#!/bin/bash
# =============================================================================
# CSP-554: AWS EMR Bootstrap Script
# Installs R 4.0.3, RStudio Server, SparkR dependencies & project packages
# Based on work by Peter Schmiedeskamp, Ido Michael & Tom Zeng
# Last updated: November 2020
# =============================================================================

set -x -e

# ── Configuration ─────────────────────────────────────────────────────────────
R_VERSION="4.0.3"
RSTUDIO_PKG="rstudio-server-rhel-1.3.1093-x86_64.rpm"
RSTUDIO_PASSWD="hadoop"

# ── Detect master node ────────────────────────────────────────────────────────
IS_MASTER=false
if grep isMaster /mnt/var/lib/info/instance.json | grep true; then
  IS_MASTER=true
fi

# ── System dependencies ───────────────────────────────────────────────────────
sudo yum install -y \
  bzip2-devel      \
  cairo-devel      \
  gcc gcc-c++ gcc-gfortran \
  libXt-devel      \
  libcurl-devel    \
  libjpeg-devel    \
  libpng-devel     \
  libtiff-devel    \
  pcre2-devel      \
  readline-devel   \
  texinfo          \
  texlive-collection-fontsrecommended

# ── Compile & install R from source ──────────────────────────────────────────
mkdir -p /tmp/R-build && cd /tmp/R-build

curl -OL "https://cran.r-project.org/src/base/R-4/R-${R_VERSION}.tar.gz"
tar -xzf "R-${R_VERSION}.tar.gz"
cd "R-${R_VERSION}"

./configure \
  --with-readline=yes          \
  --enable-R-profiling=no      \
  --enable-memory-profiling=no \
  --enable-R-shlib             \
  --with-pic                   \
  --prefix=/usr/local          \
  --with-x                     \
  --with-libpng                \
  --with-jpeglib               \
  --with-cairo                 \
  --with-recommended-packages=yes

make -j 8
sudo make install

# ── Set Hadoop/Spark environment variables for R ──────────────────────────────
cat << 'EOF' | sudo tee -a /usr/local/lib64/R/etc/Renviron
JAVA_HOME="/etc/alternatives/jre"
HADOOP_HOME_WARN_SUPPRESS="true"
HADOOP_HOME="/usr/lib/hadoop"
HADOOP_PREFIX="/usr/lib/hadoop"
HADOOP_MAPRED_HOME="/usr/lib/hadoop-mapreduce"
HADOOP_YARN_HOME="/usr/lib/hadoop-yarn"
HADOOP_COMMON_HOME="/usr/lib/hadoop"
HADOOP_HDFS_HOME="/usr/lib/hadoop-hdfs"
YARN_HOME="/usr/lib/hadoop-yarn"
HADOOP_CONF_DIR="/usr/lib/hadoop/etc/hadoop/"
YARN_CONF_DIR="/usr/lib/hadoop/etc/hadoop/"
HIVE_HOME="/usr/lib/hive"
HIVE_CONF_DIR="/usr/lib/hive/conf"
HBASE_HOME="/usr/lib/hbase"
HBASE_CONF_DIR="/usr/lib/hbase/conf"
SPARK_HOME="/usr/lib/spark"
SPARK_CONF_DIR="/usr/lib/spark/conf"
PATH=${PWD}:${PATH}
EOF

# ── Reconfigure R Java support ────────────────────────────────────────────────
sudo /usr/local/bin/R CMD javareconf

# ── Install RStudio Server (master node only) ─────────────────────────────────
if [ "$IS_MASTER" = true ]; then
  curl -OL "https://download2.rstudio.org/server/centos6/x86_64/${RSTUDIO_PKG}"
  sudo mkdir -p /etc/rstudio
  echo 'auth-minimum-user-id=100' | sudo tee -a /etc/rstudio/rserver.conf
  sudo yum install -y "${RSTUDIO_PKG}"
  sudo rstudio-server start
fi

# ── Set RStudio password for the hadoop user ──────────────────────────────────
echo "${RSTUDIO_PASSWD}" | sudo passwd hadoop --stdin

# ── Install R packages ────────────────────────────────────────────────────────
sudo /usr/local/bin/R --no-save << 'R_SCRIPT'
pkgs <- c(
  # Core Spark / data wrangling
  "sparklyr", "dplyr", "readr", "tidyr",
  # Visualisation
  "ggplot2",
  # Machine learning
  "randomForest", "caret", "MLmetrics", "e1071",
  # Utilities
  "Hmisc", "shiny", "nycflights13", "Lahman"
)
install.packages(pkgs, repos = "http://cran.rstudio.com")
R_SCRIPT

echo "Bootstrap complete."
