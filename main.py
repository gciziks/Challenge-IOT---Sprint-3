from face_recognizer import FaceRecognizer

try:
    # Inicializar sistema
    recognition_system = FaceRecognizer()
    
    # Verificar se há modelo treinado
    if not recognition_system.recognizer_trained:
        print("Nenhum modelo treinado encontrado.")
        train = input("Deseja treinar um novo usuário? (s/n): ")
        if train.lower() == 's':
            user_name = input("Digite o nome do usuário: ")
            recognition_system.collect_training_data(user_name)
    
    # Iniciar reconhecimento
    recognition_system.run_recognition()
    
except Exception as e:
    print(f"Erro fatal: {e}")
    import traceback
    traceback.print_exc()
