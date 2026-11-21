# 다른 PC에서 WSL 서버 접속 설정 기록

작성일: 2026-11-22

## 구성 주소와 포트

- 서버 PC의 LAN 주소: `<서버 PC 내부 IP>`
- 서버 PC의 외부 IP (공유기 WAN 주소): `<서버 PC 외부 IP>`
- WSL 내부 IP: `<현재_WSL_IP>` (WSL 재시작 뒤 변경될 수 있음)
- TCP 명령·상태 포트: `47090`
- 영상 포트: `47092`

## WSL IP 확인

서버 PC의 WSL에서 `<현재 WSL IP>`를 확인한다.

```bash
hostname -I
```

출력에서 현재 WSL IPv4 주소를 확인한다. WSL을 재시작해 이 주소가 바뀌면 서버 PC의 **관리자 PowerShell**에서 Windows `portproxy`의 목적지 주소를 갱신한다. 아래 `CURRENT_WSL_IP`를 방금 확인한 주소로 바꿔 실행한다.

```powershell
$wslIp = "CURRENT_WSL_IP"
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=47090
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=47092
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=47090 connectaddress=$wslIp connectport=47090
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=47092 connectaddress=$wslIp connectport=47092
netsh interface portproxy show all
```

`show all` 결과의 두 `connectaddress`가 새 WSL IP인지 확인한다. LAN 내부망용 Windows 방화벽 규칙은 같은 포트를 사용하므로 WSL IP 변경만으로 수정하지 않는다.

## WSL 서버 설정 및 실행

C 설정 파일 `~/EyetoArm/config/gui/client.json`의 `tcp.host`를 `127.0.0.1`에서 `0.0.0.0`으로 변경. 서버 시작 로그와 `ss`에서 `0.0.0.0:47090`, `0.0.0.0:47092` 수신을 확인했다.

서버 PC의 WSL에서 통합 런치를 실행한다.

```bash
cd ~/EyetoArm
source /opt/ros/jazzy/setup.bash
source ~/EyetoArm/robot_ws/install/setup.bash
ros2 launch scripts/c_stack.launch.py gui:=false start_a:=true start_moveit:=false
```

## Windows 포트 전달과 방화벽

서버 PC에서 Windows `portproxy`를 WSL로 설정했다. 확인 당시 목록:

```text
0.0.0.0:47090 -> <현재_WSL_IP>:47090
0.0.0.0:47092 -> <현재_WSL_IP>:47092
```

`iphlpsvc` 상태는 `Running`이었다. WSL IP가 바뀌면 portproxy의 대상 주소를 `<현재 WSL IP>`로 갱신해야 한다.

LAN 시험용 Windows 방화벽 인바운드 허용은 두 포트 모두 `LocalSubnet`으로 설정했다. 네트워크 프로필은 `Public`이었다. 이 규칙은 내부망만 허용하므로 외부 인터넷 클라이언트는 별도 규칙이 필요하다.

```powershell
New-NetFirewallRule -DisplayName "EyetoArm TCP 47090 LAN" -Direction Inbound -Protocol TCP -LocalPort 47090 -Action Allow -RemoteAddress LocalSubnet -Profile Any
New-NetFirewallRule -DisplayName "EyetoArm Video 47092 LAN" -Direction Inbound -Protocol TCP -LocalPort 47092 -Action Allow -RemoteAddress LocalSubnet -Profile Any
```

## 같은 공유기 내 접속 확인

다른 PC(내부 IP)에서 서버 PC 내부 IP(`<서버_PC_내부_IP>`)로 시험했다.

- TCP `47090`: `TcpTestSucceeded : True`
- TCP `47092`: `TcpTestSucceeded : True`
- 다른 PC 대시보드의 명령·상태 연결 및 영상 표시 확인
- 대시보드 재접속 후 연결 복구 확인

따라서 같은 LAN에서의 원격 대시보드·영상 연결 및 재접속은 완료했다.

## 공유기 포트포워딩

사용자가 제공한 ipTIME 설정 화면에서 다음 TCP 포워딩 항목을 확인했다.

| 외부 포트 | 내부 목적지(서버 PC 내부 IP) | 내부 포트 | 프로토콜 |
|---:|---|---:|---|
| 47090 | `<서버_PC_내부_IP>` | 47090 | TCP |
| 47092 | `<서버_PC_내부_IP>` | 47092 | TCP |

사용자가 서버 PC 외부 IP로 알려준 `<서버_PC_외부_IP>`을 접속 주소로 사용한다. 현재 공유기 WAN 주소와 일치하는지는 라우터 관리 화면에서 확인해야 한다.

## 외부망 접속 결과와 남은 작업

별도 인터넷 환경의 시험 PC에서 다음 결과가 나왔다.

```text
Test-NetConnection <서버_PC_외부_IP> -Port 47090: TcpTestSucceeded=False
Test-NetConnection <서버_PC_외부_IP> -Port 47092: TcpTestSucceeded=False
```

시험 PC 출력의 `SourceAddress`는 테스트 PC의 사설 IP였으며, 방화벽의 외부 허용 주소로 쓰려면 시험 PC의 공인 IP를 확인해야 한다. 시험 PC에서 실제 공인 IP를 확인한다.

```powershell
(Invoke-RestMethod https://api.ipify.org).Trim()
```

아래처럼 시험 PC의 공인 IP만 허용하는 인바운드 규칙을 서버 PC의 관리자 PowerShell에서 추가한 뒤 외부망 재시험이 필요하다. 이 규칙은 적용 여부가 아직 확인되지 않았다.

```powershell
New-NetFirewallRule -DisplayName "EyetoArm External 47090" -Direction Inbound -Protocol TCP -LocalPort 47090 -Action Allow -RemoteAddress <테스트_PC_공인_IP> -Profile Any
New-NetFirewallRule -DisplayName "EyetoArm External 47092" -Direction Inbound -Protocol TCP -LocalPort 47092 -Action Allow -RemoteAddress <테스트_PC_공인_IP> -Profile Any
```

외부 접속 시험은 포트포워딩·Windows 방화벽·WSL portproxy를 모두 켠 상태에서 진행한다. 외부 PC에서 각 포트를 따로 검사해 실패 구간을 구분한다.

> 주의: `47090`은 로봇 명령·상태 포트다. 인터넷에 무제한 공개하지 말고, VPN 사용 또는 시험 PC 공인 IP만 허용하는 방식으로 시험한다.




