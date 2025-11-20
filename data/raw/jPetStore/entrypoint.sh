#!/bin/bash
# filepath: d:\multimodal_microservice_extraction\data\raw\jPetStore\entrypoint.sh
set -e

echo "🚀 Starting JPetStore with HSQLDB..."

# WAR 已经在构建时解压了，直接查找 HSQLDB JAR
HSQLDB_JAR=""
if [ -f "/usr/local/tomcat/webapps/ROOT/WEB-INF/lib/hsqldb.jar" ]; then
    HSQLDB_JAR="/usr/local/tomcat/webapps/ROOT/WEB-INF/lib/hsqldb.jar"
elif [ -f "/usr/local/tomcat/webapps/ROOT/WEB-INF/lib/hsqldb-1.8.0.10.jar" ]; then
    HSQLDB_JAR="/usr/local/tomcat/webapps/ROOT/WEB-INF/lib/hsqldb-1.8.0.10.jar"
fi

if [ -n "$HSQLDB_JAR" ]; then
    echo "✅ Found HSQLDB JAR: $HSQLDB_JAR"
    
    # 创建数据库目录
    mkdir -p /usr/local/tomcat/db
    
    # 启动 HSQLDB 服务器
    echo "🔍 Starting HSQLDB database server..."
    java -cp "$HSQLDB_JAR" org.hsqldb.server.Server \
        --database.0 file:/usr/local/tomcat/db/petstore \
        --dbname.0 petstore \
        --port 9001 &
    
    echo "⏳ Waiting for HSQLDB to start..."
    sleep 5
else
    echo "⚠️ HSQLDB not found, using embedded mode"
fi

echo "✅ Starting Tomcat..."
exec catalina.sh run