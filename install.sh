#!/bin/bash
# Script de instalación automática para Laboratorio de Micro-IA Biomimética
# Maneja el error "externally-managed-environment" creando venv automáticamente

set -e  # Salir en caso de error

# Colores para terminal
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}║   🧬 Laboratorio de Micro-IA Biomimética                      ║${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}║   Instalador Automático                                       ║${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Verificar Python
echo -e "${BLUE}🔍 Verificando Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 no está instalado${NC}"
    echo -e "${YELLOW}Instala Python 3.7+ desde:${NC}"
    echo -e "  - Ubuntu/Debian: ${BLUE}sudo apt install python3 python3-venv python3-pip${NC}"
    echo -e "  - Fedora: ${BLUE}sudo dnf install python3 python3-pip${NC}"
    echo -e "  - Arch: ${BLUE}sudo pacman -S python python-pip${NC}"
    echo -e "  - Mac: ${BLUE}brew install python${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION encontrado${NC}"
echo ""

# Crear entorno virtual
echo -e "${BLUE}🔧 Creando entorno virtual...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ El entorno virtual ya existe${NC}"
    read -p "¿Deseas recrearlo? (s/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Entorno virtual recreado${NC}"
    else
        echo -e "${GREEN}✓ Usando entorno virtual existente${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Entorno virtual creado${NC}"
fi
echo ""

# Activar entorno virtual
echo -e "${BLUE}🔧 Activando entorno virtual...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Entorno virtual activado${NC}"
echo ""

# Actualizar pip
echo -e "${BLUE}📦 Actualizando pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ Pip actualizado${NC}"
echo ""

# Instalar dependencias
echo -e "${BLUE}📦 Instalando dependencias...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencias instaladas${NC}"
echo ""

# Verificar instalación
echo -e "${BLUE}🔍 Verificando instalación...${NC}"
python scripts/check_installation.py
echo ""

# Preguntar si descargar datos
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}¿Deseas descargar los datos necesarios?${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Opciones:"
echo -e "  ${BLUE}1.${NC} Solo corpus UD AnCora (básico, ~150 MB)"
echo -e "  ${BLUE}2.${NC} Corpus + datasets Hugging Face (recomendado, ~300 MB)"
echo -e "  ${BLUE}3.${NC} Todo: corpus + datasets + literatura (completo, ~500 MB)"
echo -e "  ${BLUE}0.${NC} No descargar datos (los descargarás después)"
echo ""
read -p "Selecciona una opción (0-3): " -n 1 -r
echo ""
echo ""

case $REPLY in
    1)
        echo -e "${BLUE}📥 Descargando corpus UD AnCora...${NC}"
        bash scripts/setup_corpus.sh
        echo -e "${GREEN}✓ Corpus descargado${NC}"
        ;;
    2)
        echo -e "${BLUE}📥 Descargando corpus UD AnCora...${NC}"
        bash scripts/setup_corpus.sh
        echo -e "${GREEN}✓ Corpus descargado${NC}"
        echo ""
        echo -e "${BLUE}📥 Descargando datasets de Hugging Face...${NC}"
        python scripts/download_datasets.py
        echo -e "${GREEN}✓ Datasets descargados${NC}"
        ;;
    3)
        echo -e "${BLUE}📥 Descargando corpus UD AnCora...${NC}"
        bash scripts/setup_corpus.sh
        echo -e "${GREEN}✓ Corpus descargado${NC}"
        echo ""
        echo -e "${BLUE}📥 Descargando datasets de Hugging Face...${NC}"
        python scripts/download_datasets.py
        echo -e "${GREEN}✓ Datasets descargados${NC}"
        echo ""
        echo -e "${BLUE}📥 Descargando literatura clásica...${NC}"
        python scripts/download_literature.py
        echo -e "${GREEN}✓ Literatura descargada${NC}"
        ;;
    0)
        echo -e "${YELLOW}⚠ No se descargaron datos${NC}"
        echo -e "${YELLOW}Puedes descargarlos después con:${NC}"
        echo -e "  ${BLUE}bash scripts/setup_corpus.sh${NC}"
        echo -e "  ${BLUE}python scripts/download_datasets.py${NC}"
        echo -e "  ${BLUE}python scripts/download_literature.py${NC}"
        ;;
    *)
        echo -e "${RED}❌ Opción no válida${NC}"
        ;;
esac

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Instalación completada${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Próximos pasos:${NC}"
echo ""
echo -e "1. ${BLUE}Activar el entorno virtual:${NC}"
echo -e "   ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "2. ${BLUE}Entrenar el generador:${NC}"
echo -e "   ${YELLOW}python scripts/train_final.py${NC}"
echo ""
echo -e "3. ${BLUE}Ejecutar el demo:${NC}"
echo -e "   ${YELLOW}python demo.py${NC}"
echo ""
echo -e "${BLUE}Para más información:${NC}"
echo -e "  - Tutorial completo: ${YELLOW}cat TUTORIAL.md${NC}"
echo -e "  - Preguntas frecuentes: ${YELLOW}cat FAQ.md${NC}"
echo -e "  - Documentación: ${YELLOW}cat README.md${NC}"
echo ""
