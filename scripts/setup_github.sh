#!/usr/bin/env bash
# =============================================================================
# K1 için tek seferlik GitHub kurulumu.
#   1) Repo oluşturur ve ilk commit'i main'e gönderir
#   2) 4 ekip üyesini collaborator olarak davet eder (yazma yetkisi)
#   3) CODEOWNERS üretir
#   4) Etiketleri ve her üyeye atanmış görev issue'larını açar (docs/tasks/K*.md)
#   5) main dalını korur: PR zorunlu + 1 onay + CI yeşil
#
# Gereksinim: git, GitHub CLI (gh) ve `gh auth login` yapılmış olması.
# Windows'ta Git Bash içinde çalıştırın.
# Kullanım:  cp team.env.example team.env  →  düzenle  →  bash scripts/setup_github.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f team.env ]] || { echo "team.env yok. cp team.env.example team.env ve düzenleyin."; exit 1; }
# shellcheck disable=SC1091
source team.env
command -v gh >/dev/null || { echo "GitHub CLI (gh) kurulu değil: https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Önce: gh auth login"; exit 1; }

OWNER="$(gh api user -q .login)"
FULL="$OWNER/$REPO_NAME"
echo ">> Repo: $FULL ($VISIBILITY)"
[[ "$K1" == "$OWNER" ]] || echo "!! Uyarı: team.env'deki K1 ($K1) oturum açan kullanıcıyla ($OWNER) aynı değil."

# --- 3) CODEOWNERS (ilk commit'e girsin diye önce) ---------------------------
sed -e "s/<K1>/$K1/g" -e "s/<K2>/$K2/g" -e "s/<K3>/$K3/g" -e "s/<K4>/$K4/g" -e "s/<K5>/$K5/g" \
  .github/CODEOWNERS.template > .github/CODEOWNERS

# --- 1) git + repo -------------------------------------------------------------
if [[ ! -d .git ]]; then
  git init -b main
fi
git add -A
# folds.csv ve results/ ilk commit'e girmez: gerçek PhysioNet verisiyle üretilip ayrıca commit edilir
git reset -q -- data/processed/folds.csv results/ 2>/dev/null || true
git add results/.gitkeep
git commit -m "Başlangıç: Adım 1 boru hattı, değerlendirme altyapısı, ekip iskeletleri" || true
if ! gh repo view "$FULL" >/dev/null 2>&1; then
  gh repo create "$FULL" --"$VISIBILITY" --source=. --remote=origin --push \
    --description "Yürüyüş analizi ile nörodejeneratif hastalık sınıflandırma (PhysioNet gaitndd)"
else
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$FULL.git"
  git push -u origin main
fi

# --- 2) collaborator davetleri ---------------------------------------------------
for U in "$K2" "$K3" "$K4" "$K5"; do
  echo ">> Davet: $U"
  gh api -X PUT "repos/$FULL/collaborators/$U" -f permission=push >/dev/null || echo "!! $U davet edilemedi (kullanıcı adı doğru mu?)"
done

# --- 4) etiketler + görev issue'ları -----------------------------------------------
for L in "K1:1d76db" "K2:0e8a16" "K3:fbca04" "K4:d93f0b" "K5:5319e7" "task:c5def5" "contract:b60205"; do
  gh label create "${L%%:*}" --color "${L##*:}" --repo "$FULL" --force >/dev/null
done
declare -A WHO=([K1]="$K1" [K2]="$K2" [K3]="$K3" [K4]="$K4" [K5]="$K5")
for F in docs/tasks/K*.md; do
  KEY="$(basename "$F" .md)"
  TITLE="$(head -n1 "$F" | sed 's/^# //')"
  if gh issue list --repo "$FULL" --search "in:title \"$TITLE\"" --json number -q '.[0].number' | grep -q .; then
    echo ">> Issue zaten var: $TITLE"; continue
  fi
  gh issue create --repo "$FULL" --title "$TITLE" --body-file "$F" \
    --label "$KEY" --label task --assignee "${WHO[$KEY]}" >/dev/null \
    || gh issue create --repo "$FULL" --title "$TITLE" --body-file "$F" --label "$KEY" --label task >/dev/null
  echo ">> Issue açıldı: $TITLE"
done

# --- 5) dal koruması ---------------------------------------------------------------
echo ">> main dal koruması"
gh api -X PUT "repos/$FULL/branches/main/protection" --input - >/dev/null <<JSON || \
  echo "!! Dal koruması kurulamadı. Private repoda GitHub Pro/Education gerekir; repoyu public yapın ya da Settings → Branches'tan elle kurun."
{
  "required_status_checks": {"strict": true, "contexts": ["test"]},
  "enforce_admins": false,
  "required_pull_request_reviews": {"required_approving_review_count": 1, "require_code_owner_reviews": false},
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON

echo
echo "Bitti: https://github.com/$FULL"
echo "Ekip üyeleri e-postalarındaki daveti kabul etmeli, sonra README'deki 'Kurulum' adımlarını izlemeli."
