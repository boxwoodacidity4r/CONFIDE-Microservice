FROM tomcat:9.0-jdk11-openjdk

# 安装 unzip 工具
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# 删除默认 ROOT 应用
RUN rm -rf /usr/local/tomcat/webapps/ROOT

# 复制并解压 WAR 文件
COPY jpetstore-updated.war /tmp/
RUN mkdir -p /usr/local/tomcat/webapps/ROOT && \
    cd /usr/local/tomcat/webapps/ROOT && \
    unzip -q /tmp/jpetstore-updated.war && \
    rm /tmp/jpetstore-updated.war

# 创建数据库目录
RUN mkdir -p /usr/local/tomcat/db

# 复制 entrypoint 脚本
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 暴露端口
EXPOSE 8080 9001

# 使用 entrypoint 脚本启动
ENTRYPOINT ["/entrypoint.sh"]