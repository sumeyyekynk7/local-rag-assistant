from foundry_local_sdk import Configuration, FoundryLocalManager


def main() -> None:
    config = Configuration(app_name="local-rag-assistant")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    models = manager.catalog.list_models()

    print(f"Kullanılabilir model sayısı: {len(models)}\n")

    for model in models:
        print(f"Alias: {model.alias}")
        print(f"ID: {model.id}")
        print(f"Yetenekler: {model.capabilities}")
        print(f"Girdi türleri: {model.input_modalities}")
        print(f"Çıktı türleri: {model.output_modalities}")
        print(f"Önbellekte mi: {model.is_cached}")
        print("-" * 60)


if __name__ == "__main__":
    main()