# 服务器端配置ss-libev

安装出问题参考: [vultr登上谷歌学术](http://fpcsongazure.top/how-to-fuck-gfw-to-get-a-paper/)

### 安装ss-libev

```shell
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:max-c-lv/shadowsocks-libev -y
sudo apt-get update
sudo apt install shadowsocks-libev
```

### 配置ss-libev

ip设置成 ["::","0.0.0.0"]是ipv4和ipv6都可以连接的版本.



### 启动ss-libev

```shell
ss-server -c config.json
```





# ubuntu本地配置ss-libev





### 安装本地ss-libev

```shell
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:max-c-lv/shadowsocks-libev -y
sudo apt-get update
sudo apt install shadowsocks-libev
```



## 配置ss config.json文件

本地随意保存一个位置即可,配置好port, ip, method, password



### 运行本地ss

默认浏览器已经配置好. 

```shell
ss-local -u -c config.json
```

config.json为配置文件的地址. 也可以将其加入到启动项中,开机自动启动.



### 命令行走ipv6设置

使用proxychains.

首先安装:

```shell
sudo apt-get install proxychains4
```



配置文件:

```shell
sudo vim /etc/proxychains.conf
```



将`socks4 127.0.0.1 9095`更改为`socks4 127.0.0.1 1080`即可.



使用方法:

在需要的命令行前加上proxychains4,比如:

```shell
proxychains4 wget http://....
```





