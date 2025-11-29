pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        VENV_DIR = "${WORKSPACE}/venv"
    }
    
    stages {
        stage('Environment Setup') {
            steps {
                echo '================================'
                echo 'Stage 1: Environment Setup'
                echo '================================'
                
                script {
                    // Checkout code
                    checkout scm
                    
                    // Create virtual environment and install dependencies
                    sh '''
                        python3 --version || python --version
                        python3 -m venv ${VENV_DIR} || python -m venv ${VENV_DIR}
                        source ${VENV_DIR}/bin/activate || . ${VENV_DIR}/Scripts/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                        echo "✓ Environment setup completed"
                    '''
                }
            }
        }
        
        stage('Pipeline Compilation') {
            steps {
                echo '================================'
                echo 'Stage 2: Pipeline Compilation'
                echo '================================'
                
                script {
                    sh '''
                        source ${VENV_DIR}/bin/activate || . ${VENV_DIR}/Scripts/activate
                        
                        # Compile MLflow components to YAML (Kubeflow alternative)
                        echo "Compiling components..."
                        python3 compile_components.py || python compile_components.py
                        
                        # Compile MLflow pipeline definition
                        echo "Compiling pipeline..."
                        python3 pipeline.py compile || python pipeline.py compile
                        
                        # Verify pipeline.yaml exists
                        if [ -f pipeline.yaml ]; then
                            echo "✓ Pipeline compiled successfully: pipeline.yaml"
                            ls -lh pipeline.yaml
                        else
                            echo "✗ ERROR: pipeline.yaml not found!"
                            exit 1
                        fi
                        
                        # Verify component YAML files exist
                        if [ -d components ] && [ "$(ls -A components/*.yaml 2>/dev/null)" ]; then
                            echo "✓ Component YAML files found:"
                            ls -lh components/*.yaml
                        else
                            echo "✗ ERROR: Component YAML files not found!"
                            exit 1
                        fi
                    '''
                }
            }
        }
        
        stage('Code Quality & Testing') {
            steps {
                echo '================================'
                echo 'Stage 3: Code Quality & Testing'
                echo '================================'
                
                script {
                    sh '''
                        source ${VENV_DIR}/bin/activate || . ${VENV_DIR}/Scripts/activate
                        
                        # Check if all Python files are syntactically correct
                        python3 -m py_compile src/*.py pipeline.py compile_components.py || \
                        python -m py_compile src/*.py pipeline.py compile_components.py
                        echo "✓ All Python files compiled successfully"
                    '''
                }
            }
        }
    }
    
    post {
        always {
            echo '================================'
            echo 'Pipeline Execution Completed'
            echo '================================'
            
            // Archive artifacts
            archiveArtifacts artifacts: 'pipeline.yaml', allowEmptyArchive: true
            archiveArtifacts artifacts: 'components/**/*.yaml', allowEmptyArchive: true
            
            // Clean up (optional - comment out if you want to keep workspace)
            // cleanWs()
        }
        
        success {
            echo '✓ Pipeline compiled successfully!'
        }
        
        failure {
            echo '✗ Pipeline compilation failed!'
        }
    }
}