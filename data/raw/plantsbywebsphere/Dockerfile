FROM openliberty/open-liberty:full-java11-openj9-ubi

# 复制 Liberty 服务器配置和应用
COPY --chown=1001:0 build/wlp/usr/servers/PlantsByWebSphereServer/ /config/

# 暴露端口
EXPOSE 9080 9443