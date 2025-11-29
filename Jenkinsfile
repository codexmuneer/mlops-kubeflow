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
                    
                    // Create virtual environment
                    sh '''
                        python3 -m venv ${VENV_DIR}
                        . ${VENV_DIR}/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
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
                        . ${VENV_DIR}/bin/activate
                        
                        # Compile MLflow components to YAML (Kubeflow alternative)
                        python3 compile_components.py
                        
                        # Compile MLflow pipeline definition
                        python3 pipeline.py compile
                        
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
                        . ${VENV_DIR}/bin/activate
                        
                        # Check if all Python files are syntactically correct
                        python3 -m py_compile src/*.py pipeline.py compile_components.py
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
            
            // Clean up
            cleanWs()
        }
        
        success {
            echo '✓ Pipeline compiled successfully!'
        }
        
        failure {
            echo '✗ Pipeline compilation failed!'
        }
    }
}