# 使用 Open Liberty 基础镜像 (带 JDK8 OpenJ9)
# 使用带 Web Profile 的 Open Liberty 基础镜像（包含 JSP 支持）
FROM openliberty/open-liberty:25.0.0.3-full-java8-ibmjava-ubi-amd64


WORKDIR /config

# 复制构建好的 war 文件
COPY acmeair-webapp/build/libs/acmeair-webapp-2.0.0-SNAPSHOT.war /config/dropins/acmeair.war

EXPOSE 9080


# 不需要 CMD，Open Liberty 会自动启动
