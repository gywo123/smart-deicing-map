# GitHub Push 규칙 (smart-deicing-map)

## 원격 저장소
- 기본 원격 저장소: `git@github.com:gywo123/smart-deicing-map.git`

## 브랜치 선택 규칙
- **완성/안정 상태**: `main` 브랜치에 push
- **개발 진행 중**: `DEV` 브랜치에 push

## Push 전 체크리스트
1. 현재 작업 상태 확인
   - 완료 수준이면 `main`
   - 진행 중이면 `DEV`
2. 변경 파일 확인
   - `git status`
3. 커밋 생성
   - `git add .`
   - `git commit -m "<메시지>"`
4. 대상 브랜치로 push
   - `git push origin main`
   - 또는 `git push origin DEV`

## 보안 규칙 (중요)
- GitHub 토큰은 **저장소 파일에 절대 기록하지 않음**.
- 토큰은 로컬 환경변수/OS 자격증명 저장소로만 관리.
- 실수로 토큰이 커밋되면 즉시 토큰 폐기 후 재발급.

## 빠른 명령 예시
```bash
# 원격 등록(최초 1회)
git remote add origin git@github.com:gywo123/smart-deicing-map.git

# 진행 중 작업 푸시
git checkout -B DEV
git add .
git commit -m "wip: 작업 내용"
git push -u origin DEV

# 완료 작업 푸시
git checkout -B main
git add .
git commit -m "feat: 완료 내용"
git push -u origin main
```

---
이 파일은 push 시 참고용 운영 규칙 문서입니다.
